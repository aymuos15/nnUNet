"""
Specialized planner implementations for specific use cases.

This module contains planner variants that use alternative resampling strategies
or preprocessing approaches compared to the standard planners.
"""

from typing import Union, List, Tuple

from nnunetv2.experiment_planning.config.defaults import DEFAULT_ANISO_THRESHOLD
from nnunetv2.experiment_planning.planners.standard import ExperimentPlanner
from nnunetv2.experiment_planning.planners.residual_unet import nnUNetPlannerResEncL
from nnunetv2.preprocessing.resampling.no_resampling import no_resampling_hack
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet


# ============================================================================
# No Resampling Planners
# ============================================================================


class nnUNetPlannerResEncL_noResampling(nnUNetPlannerResEncL):
    """
    This planner will generate 3d_lowres as well. Don't trust it. Everything will remain in the original shape.
    No resampling will ever be done.
    """
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlans_noResampling',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + '_' + configuration_name

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        resampling_data = no_resampling_hack
        resampling_data_kwargs = {}
        resampling_seg = no_resampling_hack
        resampling_seg_kwargs = {}
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

        determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
        functions for each configuration

        """
        resampling_fn = no_resampling_hack
        resampling_fn_kwargs = {}
        return resampling_fn, resampling_fn_kwargs


# ============================================================================
# Torch Resampling Planners
# ============================================================================


class nnUNetPlannerResEncL_torchres(nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlans_torchres',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + '_' + configuration_name

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        resampling_data = resample_torch_fornnunet
        resampling_data_kwargs = {
            "is_seg": False,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        resampling_seg = resample_torch_fornnunet
        resampling_seg_kwargs = {
            "is_seg": True,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

        determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
        functions for each configuration

        """
        resampling_fn = resample_torch_fornnunet
        resampling_fn_kwargs = {
            "is_seg": False,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        return resampling_fn, resampling_fn_kwargs


class nnUNetPlannerResEncL_torchres_sepz(nnUNetPlannerResEncL):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 24,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetResEncUNetLPlans_torchres_sepz',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + '_' + configuration_name

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        resampling_data = resample_torch_fornnunet
        resampling_data_kwargs = {
            "is_seg": False,
            'force_separate_z': None,
            'memefficient_seg_resampling': False,
            'separate_z_anisotropy_threshold': DEFAULT_ANISO_THRESHOLD
        }
        resampling_seg = resample_torch_fornnunet
        resampling_seg_kwargs = {
            "is_seg": True,
            'force_separate_z': None,
            'memefficient_seg_resampling': False,
            'separate_z_anisotropy_threshold': DEFAULT_ANISO_THRESHOLD
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

        determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
        functions for each configuration

        """
        resampling_fn = resample_torch_fornnunet
        resampling_fn_kwargs = {
            "is_seg": False,
            'force_separate_z': None,
            'memefficient_seg_resampling': False,
            'separate_z_anisotropy_threshold': DEFAULT_ANISO_THRESHOLD
        }
        return resampling_fn, resampling_fn_kwargs


class nnUNetPlanner_torchres(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetPlans_torchres',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False):
        super().__init__(dataset_name_or_id, gpu_memory_target_in_gb, preprocessor_name, plans_name,
                         overwrite_target_spacing, suppress_transpose)

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + '_' + configuration_name

    def determine_resampling(self, *args, **kwargs):
        """
        returns what functions to use for resampling data and seg, respectively. Also returns kwargs
        resampling function must be callable(data, current_spacing, new_spacing, **kwargs)

        determine_resampling is called within get_plans_for_configuration to allow for different functions for each
        configuration
        """
        resampling_data = resample_torch_fornnunet
        resampling_data_kwargs = {
            "is_seg": False,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        resampling_seg = resample_torch_fornnunet
        resampling_seg_kwargs = {
            "is_seg": True,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        return resampling_data, resampling_data_kwargs, resampling_seg, resampling_seg_kwargs

    def determine_segmentation_softmax_export_fn(self, *args, **kwargs):
        """
        function must be callable(data, new_shape, current_spacing, new_spacing, **kwargs). The new_shape should be
        used as target. current_spacing and new_spacing are merely there in case we want to use it somehow

        determine_segmentation_softmax_export_fn is called within get_plans_for_configuration to allow for different
        functions for each configuration

        """
        resampling_fn = resample_torch_fornnunet
        resampling_fn_kwargs = {
            "is_seg": False,
            'force_separate_z': False,
            'memefficient_seg_resampling': False
        }
        return resampling_fn, resampling_fn_kwargs
