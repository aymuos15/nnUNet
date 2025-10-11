import warnings

import numpy as np
from typing import Union, List, Tuple

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
from torch import nn

from nnunetv2.experiment_planning.planners.standard import ExperimentPlanner

from nnunetv2.experiment_planning.planners.components.network_topology import get_pool_and_conv_props
from nnunetv2.experiment_planning.planners.components.architecture_config import (
    build_architecture_kwargs,
    compute_features_per_stage
)
from nnunetv2.experiment_planning.planners.components.patch_size_planning import (
    compute_initial_patch_size,
    reduce_patch_size_step
)
from nnunetv2.experiment_planning.planners.components.plan_builder import build_plan_dict


class ResEncUNetPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNet
        # the following two numbers are really arbitrary and were set to reproduce default nnU-Net's configurations as
        # much as possible
        self.UNet_reference_val_3d = 680000000
        self.UNet_reference_val_2d = 135000000
        self.UNet_blocks_per_stage_encoder = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
        self.UNet_blocks_per_stage_decoder = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        if configuration_name == '2d' or configuration_name == '3d_fullres':
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return 'nnUNetPlans' + '_' + configuration_name
        else:
            return self.plans_identifier + '_' + configuration_name

    def get_plans_for_configuration(self,
                                    spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
                                    median_shape: Union[np.ndarray, Tuple[int, ...]],
                                    data_identifier: str,
                                    approximate_n_voxels_dataset: float,
                                    _cache: dict) -> dict:
        def _keygen(patch_size, strides):
            return str(patch_size) + '_' + str(strides)

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"
        num_input_channels = len(self.dataset_json['channel_names'].keys()
                                 if 'channel_names' in self.dataset_json.keys()
                                 else self.dataset_json['modality'].keys())
        max_num_features = self.UNet_max_features_2d if len(spacing) == 2 else self.UNet_max_features_3d

        # find an initial patch size
        initial_patch_size = compute_initial_patch_size(spacing, median_shape)

        # use that to get the network topology. Note that this changes the patch_size depending on the number of
        # pooling operations (must be divisible by 2**num_pool in each axis)
        network_num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, patch_size, \
        shape_must_be_divisible_by = get_pool_and_conv_props(spacing, initial_patch_size,
                                                             self.UNet_featuremap_min_edge_length,
                                                             999999)
        num_stages = len(pool_op_kernel_sizes)

        # ResEncUNet uses n_blocks_per_stage instead of n_conv_per_stage, so we build architecture kwargs manually
        # rather than using the utility (which is designed for PlainConvUNet)
        from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
        unet_conv_op = convert_dim_to_conv_op(len(spacing))
        norm = get_matching_instancenorm(unet_conv_op)
        architecture_kwargs = {
            'network_class_name': self.UNet_class.__module__ + '.' + self.UNet_class.__name__,
            'arch_kwargs': {
                'n_stages': num_stages,
                'features_per_stage': compute_features_per_stage(num_stages, max_num_features, self.UNet_base_num_features),
                'conv_op': unet_conv_op.__module__ + '.' + unet_conv_op.__name__,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'n_blocks_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
                'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
                'conv_bias': True,
                'norm_op': norm.__module__ + '.' + norm.__name__,
                'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': None,
                'dropout_op_kwargs': None,
                'nonlin': 'torch.nn.LeakyReLU',
                'nonlin_kwargs': {'inplace': True},
            },
            '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin'),
        }

        # now estimate vram consumption
        if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
            estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
        else:
            estimate = self.static_estimate_VRAM_usage(patch_size,
                                                       num_input_channels,
                                                       len(self.dataset_json['labels'].keys()),
                                                       architecture_kwargs['network_class_name'],
                                                       architecture_kwargs['arch_kwargs'],
                                                       architecture_kwargs['_kw_requires_import'],
                                                       )
            _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # how large is the reference for us here (batch size etc)?
        # adapt for our vram target
        reference = (self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d) * \
                    (self.UNet_vram_target_GB / self.UNet_reference_val_corresp_GB)

        while estimate > reference:
            # Reduce patch size
            patch_size, pool_op_kernel_sizes, conv_kernel_sizes, shape_must_be_divisible_by = \
                reduce_patch_size_step(patch_size, spacing, median_shape, shape_must_be_divisible_by,
                                      self.UNet_featuremap_min_edge_length)

            num_stages = len(pool_op_kernel_sizes)
            architecture_kwargs['arch_kwargs'].update({
                'n_stages': num_stages,
                'kernel_sizes': conv_kernel_sizes,
                'strides': pool_op_kernel_sizes,
                'features_per_stage': compute_features_per_stage(num_stages, max_num_features, self.UNet_base_num_features),
                'n_blocks_per_stage': self.UNet_blocks_per_stage_encoder[:num_stages],
                'n_conv_per_stage_decoder': self.UNet_blocks_per_stage_decoder[:num_stages - 1],
            })
            if _keygen(patch_size, pool_op_kernel_sizes) in _cache.keys():
                estimate = _cache[_keygen(patch_size, pool_op_kernel_sizes)]
            else:
                estimate = self.static_estimate_VRAM_usage(
                    patch_size,
                    num_input_channels,
                    len(self.dataset_json['labels'].keys()),
                    architecture_kwargs['network_class_name'],
                    architecture_kwargs['arch_kwargs'],
                    architecture_kwargs['_kw_requires_import'],
                )
                _cache[_keygen(patch_size, pool_op_kernel_sizes)] = estimate

        # alright now let's determine the batch size. This will give self.UNet_min_batch_size if the while loop was
        # executed. If not, additional vram headroom is used to increase batch size
        from nnunetv2.experiment_planning.planners.components.vram_estimation import compute_batch_size
        ref_bs = self.UNet_reference_val_corresp_bs_2d if len(spacing) == 2 else self.UNet_reference_val_corresp_bs_3d
        reference_val = self.UNet_reference_val_2d if len(spacing) == 2 else self.UNet_reference_val_3d
        batch_size = compute_batch_size(
            estimate,
            reference_val,
            ref_bs,
            self.UNet_vram_target_GB,
            self.UNet_reference_val_corresp_GB,
            self.UNet_min_batch_size,
            patch_size,
            approximate_n_voxels_dataset,
            self.max_dataset_covered
        )

        resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = self.determine_segmentation_softmax_export_fn()

        normalization_schemes, mask_is_used_for_norm = \
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()

        plan = build_plan_dict(
            data_identifier,
            self.preprocessor_name,
            batch_size,
            patch_size,
            median_shape,
            spacing,
            normalization_schemes,
            mask_is_used_for_norm,
            resampling_data,
            resampling_seg,
            resampling_data_kwargs,
            resampling_seg_kwargs,
            resampling_softmax,
            resampling_softmax_kwargs,
            architecture_kwargs
        )
        return plan


class nnUNetPlannerResEncM(ResEncUNetPlanner):
    """
    Target is ~9-11 GB VRAM max -> older Titan, RTX 2080ti
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetMPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        if gpu_memory_target_in_gb != 8:
            warnings.warn("WARNING: You are running nnUNetPlannerM with a non-standard gpu_memory_target_in_gb. "
                          f"Expected 8, got {gpu_memory_target_in_gb}."
                          "You should only see this warning if you modified this value intentionally!!")
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNet

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 8

        # this is supposed to give the same GPU memory requirement as the default nnU-Net
        self.UNet_reference_val_3d = 680000000
        self.UNet_reference_val_2d = 135000000
        self.max_dataset_covered = 1


class nnUNetPlannerResEncL(ResEncUNetPlanner):
    """
    Target is ~24 GB VRAM max -> RTX 4090, Titan RTX, Quadro 6000
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        if gpu_memory_target_in_gb != 24:
            warnings.warn("WARNING: You are running nnUNetPlannerL with a non-standard gpu_memory_target_in_gb. "
                          f"Expected 24, got {gpu_memory_target_in_gb}."
                          "You should only see this warning if you modified this value intentionally!!")
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNet

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 24

        self.UNet_reference_val_3d = 2100000000  # 1840000000
        self.UNet_reference_val_2d = 380000000  # 352666667
        self.max_dataset_covered = 1


class nnUNetPlannerResEncXL(ResEncUNetPlanner):
    """
    Target is 40 GB VRAM max -> A100 40GB, RTX 6000 Ada Generation
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 40,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetXLPlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        if gpu_memory_target_in_gb != 40:
            warnings.warn("WARNING: You are running nnUNetPlannerXL with a non-standard gpu_memory_target_in_gb. "
                          f"Expected 40, got {gpu_memory_target_in_gb}."
                          "You should only see this warning if you modified this value intentionally!!")
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)
        self.UNet_class = ResidualEncoderUNet

        self.UNet_vram_target_GB = gpu_memory_target_in_gb
        self.UNet_reference_val_corresp_GB = 40

        self.UNet_reference_val_3d = 3600000000
        self.UNet_reference_val_2d = 560000000
        self.max_dataset_covered = 1


if __name__ == '__main__':
    # we know both of these networks run with batch size 2 and 12 on ~8-10GB, respectively
    net = ResidualEncoderUNet(input_channels=1, n_stages=6, features_per_stage=(32, 64, 128, 256, 320, 320),
                              conv_op=nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2),
                              n_blocks_per_stage=(1, 3, 4, 6, 6, 6), num_classes=3,
                              n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
                              conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs={}, dropout_op=None,
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True)
    print(net.compute_conv_feature_map_size((128, 128, 128)))  # -> 558319104. The value you see above was finetuned
    # from this one to match the regular nnunetplans more closely

    net = ResidualEncoderUNet(input_channels=1, n_stages=7, features_per_stage=(32, 64, 128, 256, 512, 512, 512),
                              conv_op=nn.Conv2d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2, 2),
                              n_blocks_per_stage=(1, 3, 4, 6, 6, 6, 6), num_classes=3,
                              n_conv_per_stage_decoder=(1, 1, 1, 1, 1, 1),
                              conv_bias=True, norm_op=nn.InstanceNorm2d, norm_op_kwargs={}, dropout_op=None,
                              nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=True)
    print(net.compute_conv_feature_map_size((512, 512)))  # -> 129793792
