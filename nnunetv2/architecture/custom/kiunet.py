"""
Dynamic KiU-Net Architecture

This module implements a dynamic version of KiU-Net (Kim et al.) that is compatible
with nnU-Net's planning and configuration system.

KiU-Net combines two complementary branches:
- U-Net branch: Uses downsampling (standard encoder-decoder)
- Ki-Net branch: Uses upsampling in encoder (overcomplete representation)
- Cross-Refinement Blocks (CRFBs): Exchange features between branches

Reference: https://github.com/jeya-maria-jose/KiU-Net-pytorch

The dynamic version supports:
- Configurable number of stages
- Configurable feature channels per stage
- 2D and 3D operations via parameterized conv_op
- Deep supervision
- nnU-Net's standard interface
"""

from typing import Type, Union, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp, maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


class CRFB(nn.Module):
    """
    Cross-Refinement Block (CRFB)

    Exchanges features between U-Net and Ki-Net branches at each stage.
    Uses interpolation to handle spatial dimension mismatches.
    """
    def __init__(
        self,
        unet_channels: int,
        kinet_channels: int,
        conv_op: Type[_ConvNd],
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        interpolation_mode: str = 'bilinear',
    ):
        super().__init__()

        self.unet_channels = unet_channels
        self.kinet_channels = kinet_channels
        self.interpolation_mode = interpolation_mode

        # 1x1 convolutions for cross-branch feature adaptation
        self.unet_to_kinet = conv_op(unet_channels, kinet_channels, 1, 1, 0, bias=True)
        self.kinet_to_unet = conv_op(kinet_channels, unet_channels, 1, 1, 0, bias=True)

        # Optional normalization
        if norm_op is not None:
            norm_op_kwargs = norm_op_kwargs if norm_op_kwargs is not None else {}
            self.unet_norm = norm_op(unet_channels, **norm_op_kwargs)
            self.kinet_norm = norm_op(kinet_channels, **norm_op_kwargs)
        else:
            self.unet_norm = None
            self.kinet_norm = None

    def forward(self, unet_features: torch.Tensor, kinet_features: torch.Tensor):
        """
        Args:
            unet_features: Features from U-Net branch (B, C_unet, H, W, [D])
            kinet_features: Features from Ki-Net branch (B, C_kinet, H', W', [D'])

        Returns:
            Tuple of refined features (unet_refined, kinet_refined)
        """
        # Get target sizes for interpolation
        unet_size = unet_features.shape[2:]
        kinet_size = kinet_features.shape[2:]

        # Interpolate to match spatial dimensions
        if unet_size != kinet_size:
            # Upsample unet features to match kinet size (kinet is typically larger)
            unet_to_kinet_size = F.interpolate(
                unet_features,
                size=kinet_size,
                mode=self.interpolation_mode,
                align_corners=False if self.interpolation_mode != 'nearest' else None
            )

            # Downsample kinet features to match unet size
            kinet_to_unet_size = F.interpolate(
                kinet_features,
                size=unet_size,
                mode=self.interpolation_mode,
                align_corners=False if self.interpolation_mode != 'nearest' else None
            )
        else:
            unet_to_kinet_size = unet_features
            kinet_to_unet_size = kinet_features

        # Cross-branch feature transformation
        unet_adapted = self.unet_to_kinet(unet_to_kinet_size)
        kinet_adapted = self.kinet_to_unet(kinet_to_unet_size)

        # Residual refinement
        unet_refined = unet_features + kinet_adapted
        kinet_refined = kinet_features + unet_adapted

        # Optional normalization
        if self.unet_norm is not None:
            unet_refined = self.unet_norm(unet_refined)
        if self.kinet_norm is not None:
            kinet_refined = self.kinet_norm(kinet_refined)

        return unet_refined, kinet_refined


class DynamicKiUNet(nn.Module):
    """
    Dynamic KiU-Net compatible with nnU-Net's planning system.

    This architecture combines two complementary pathways:
    1. U-Net branch: Standard encoder-decoder with downsampling
    2. Ki-Net branch: Inverted encoder with upsampling (overcomplete)

    Cross-Refinement Blocks (CRFBs) exchange information between branches
    at each stage, allowing both local and global feature learning.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        pool_type: str = 'conv',
    ):
        """
        Initialize Dynamic KiU-Net.

        Args:
            input_channels: Number of input channels
            num_classes: Number of output segmentation classes
            n_stages: Number of encoder/decoder stages
            features_per_stage: Feature channels at each stage
            conv_op: Convolution operation (Conv2d, Conv3d, etc.)
            kernel_sizes: Kernel sizes per stage
            strides: Strides per stage (for downsampling in U-Net branch)
            n_conv_per_stage: Number of convolutions per encoder stage
            n_conv_per_stage_decoder: Number of convolutions per decoder stage
            conv_bias: Whether to use bias in convolutions
            norm_op: Normalization operation (InstanceNorm, BatchNorm, etc.)
            norm_op_kwargs: Kwargs for normalization
            dropout_op: Dropout operation
            dropout_op_kwargs: Kwargs for dropout
            nonlin: Non-linearity (ReLU, LeakyReLU, etc.)
            nonlin_kwargs: Kwargs for non-linearity
            deep_supervision: Whether to use deep supervision
            nonlin_first: Whether to apply nonlin before or after norm
            pool_type: Type of pooling for U-Net encoder ('conv', 'max', or 'avg')
                      'conv' uses strided convolution, 'max'/'avg' use explicit pooling
        """
        super().__init__()

        # Normalize inputs to lists
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        # Store configuration
        self.n_stages = n_stages
        self.features_per_stage = features_per_stage
        self.conv_op = conv_op
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.pool_type = pool_type
        self.strides = strides

        # Add decoder wrapper for nnU-Net compatibility
        # nnU-Net's set_deep_supervision_enabled expects model.decoder.deep_supervision
        class DecoderWrapper:
            def __init__(self, parent):
                self._parent = parent

            @property
            def deep_supervision(self):
                return self._parent.deep_supervision

            @deep_supervision.setter
            def deep_supervision(self, value):
                self._parent.deep_supervision = value

        self.decoder = DecoderWrapper(self)

        # Determine interpolation mode based on conv dimension
        if conv_op == nn.Conv2d:
            self.interpolation_mode = 'bilinear'
        elif conv_op == nn.Conv3d:
            self.interpolation_mode = 'trilinear'
        else:
            self.interpolation_mode = 'nearest'

        # Build U-Net Encoder (standard downsampling path)
        # Matches original KiU-Net: uses explicit pooling when pool_type='max'/'avg'
        self.unet_encoder = nn.ModuleList()
        self.unet_pools = nn.ModuleList() if pool_type in ['max', 'avg'] else None
        unet_in_channels = input_channels

        for stage_idx in range(n_stages):
            stage_modules = []

            # Add pooling layer first if using explicit pooling (and not first stage)
            if pool_type in ['max', 'avg'] and stage_idx > 0:
                stride = strides[stage_idx]
                # Check if stride requires pooling
                if (isinstance(stride, int) and stride != 1) or \
                   (isinstance(stride, (tuple, list)) and any(s != 1 for s in stride)):
                    pool_op = get_matching_pool_op(conv_op, pool_type=pool_type)
                    stage_modules.append(pool_op(kernel_size=stride, stride=stride))

            # Determine conv stride based on pooling type
            conv_stride = 1 if pool_type in ['max', 'avg'] else strides[stage_idx]

            # Add conv blocks
            stage_modules.append(
                StackedConvBlocks(
                    n_conv_per_stage[stage_idx],
                    conv_op,
                    unet_in_channels,
                    features_per_stage[stage_idx],
                    kernel_sizes[stage_idx],
                    conv_stride,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first
                )
            )

            self.unet_encoder.append(nn.Sequential(*stage_modules))
            unet_in_channels = features_per_stage[stage_idx]

        # Build Ki-Net Encoder (upsampling path - inverse of U-Net)
        # Ki-Net processes features at progressively larger spatial scales
        self.kinet_encoder = nn.ModuleList()
        kinet_in_channels = input_channels
        # Ki-Net uses reversed feature progression
        kinet_features_per_stage = list(reversed(features_per_stage))
        for stage_idx in range(n_stages):
            stage = StackedConvBlocks(
                n_conv_per_stage[stage_idx],
                conv_op,
                kinet_in_channels,
                kinet_features_per_stage[stage_idx],
                kernel_sizes[stage_idx],
                1,  # No stride, use interpolation instead
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            )
            self.kinet_encoder.append(stage)
            kinet_in_channels = kinet_features_per_stage[stage_idx]

        # Build Cross-Refinement Blocks (CRFBs) for encoder stages
        self.crfb_encoder = nn.ModuleList()
        for stage_idx in range(n_stages):
            crfb = CRFB(
                unet_channels=features_per_stage[stage_idx],
                kinet_channels=kinet_features_per_stage[stage_idx],
                conv_op=conv_op,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                interpolation_mode=self.interpolation_mode,
            )
            self.crfb_encoder.append(crfb)

        # Build U-Net Decoder (standard upsampling path)
        self.unet_decoder = nn.ModuleList()
        self.unet_transpconvs = nn.ModuleList()
        transpconv_op = get_matching_convtransp(conv_op=conv_op)

        for stage_idx in range(n_stages - 1, 0, -1):
            # Transpose convolution for upsampling
            input_features_below = features_per_stage[stage_idx]
            input_features_skip = features_per_stage[stage_idx - 1]
            stride = strides[stage_idx]

            self.unet_transpconvs.append(
                transpconv_op(
                    input_features_below,
                    input_features_skip,
                    stride,
                    stride,
                    bias=conv_bias
                )
            )

            # Decoder stage (processes concatenated features)
            decoder_stage = StackedConvBlocks(
                n_conv_per_stage_decoder[stage_idx - 1],
                conv_op,
                2 * input_features_skip,  # Concatenation: skip + upsampled
                input_features_skip,
                kernel_sizes[stage_idx - 1],
                1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            )
            self.unet_decoder.append(decoder_stage)

        # Build Ki-Net Decoder (downsampling path - inverse of Ki-Net encoder)
        # Ki-Net decoder goes from smallest feature channels (end of encoder) back to largest
        # Starts from kinet_features_per_stage[-1] and goes back to kinet_features_per_stage[0]
        self.kinet_decoder = nn.ModuleList()
        self.kinet_downsample = nn.ModuleList()

        for stage_idx in range(n_stages - 1, 0, -1):
            # We're going from stage_idx to stage_idx-1
            # Current stage has kinet_features_per_stage[stage_idx] channels
            # Next stage should have kinet_features_per_stage[stage_idx-1] channels
            input_features = kinet_features_per_stage[stage_idx]
            output_features = kinet_features_per_stage[stage_idx - 1]

            # Use strided convolution for downsampling
            # We need to downsample by the stride that was used in encoder at stage_idx
            self.kinet_downsample.append(
                conv_op(
                    input_features,
                    output_features,
                    kernel_sizes[stage_idx],
                    strides[stage_idx] if stage_idx < len(strides) else 2,
                    padding=1,
                    bias=conv_bias
                )
            )

            # Decoder stage processes concatenated features
            decoder_stage = StackedConvBlocks(
                n_conv_per_stage_decoder[stage_idx - 1],
                conv_op,
                2 * output_features,  # Concatenation of downsampled + skip
                output_features,
                kernel_sizes[stage_idx - 1],
                1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            )
            self.kinet_decoder.append(decoder_stage)

        # Build CRFBs for decoder stages
        self.crfb_decoder = nn.ModuleList()
        for stage_idx in range(n_stages - 1):
            crfb = CRFB(
                unet_channels=features_per_stage[stage_idx],
                kinet_channels=kinet_features_per_stage[n_stages - 1 - stage_idx],
                conv_op=conv_op,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                interpolation_mode=self.interpolation_mode,
            )
            self.crfb_decoder.append(crfb)

        # Final segmentation heads
        # Fusion of both branches at the output
        # U-Net decoder ends at features_per_stage[0] channels
        # Ki-Net decoder ends at kinet_features_per_stage[0] channels (after going through all stages)
        self.unet_seg_head = conv_op(features_per_stage[0], num_classes, 1, 1, 0, bias=True)
        self.kinet_seg_head = conv_op(kinet_features_per_stage[0], num_classes, 1, 1, 0, bias=True)

        # Deep supervision heads (optional)
        # These heads are applied to decoder outputs at each stage
        # We skip the last decoder stage (it's used for the fused final output)
        # So we only create heads for intermediate decoder stages
        if deep_supervision:
            self.unet_deep_supervision_heads = nn.ModuleList()
            # Create heads for intermediate decoder stages only (skip the last one)
            # With n_stages=4: decoder has 3 stages (0,1,2), we create heads for 0,1 only
            for dec_idx in range(n_stages - 2):
                # Decoder at dec_idx outputs features_per_stage[n_stages - 2 - dec_idx] channels
                out_channels = features_per_stage[n_stages - 2 - dec_idx]
                self.unet_deep_supervision_heads.append(
                    conv_op(out_channels, num_classes, 1, 1, 0, bias=True)
                )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through Dynamic KiU-Net.

        Args:
            x: Input tensor (B, C, H, W, [D])

        Returns:
            If deep_supervision: List of outputs at different scales
            Else: Single output tensor
        """
        # Store encoder features for skip connections
        unet_skips = []
        kinet_skips = []

        # === ENCODER ===
        unet_features = x
        kinet_features = x
        input_spatial_size = x.shape[2:]  # Store input size

        for stage_idx in range(self.n_stages):
            # U-Net encoder stage (with downsampling)
            unet_features = self.unet_encoder[stage_idx](unet_features)

            # Ki-Net encoder stage (matches original: Conv → Interpolate)
            # Process convolutions first, then upsample (original KiU-Net order)
            kinet_features = self.kinet_encoder[stage_idx](kinet_features)

            # Upsample after conv (with memory limit to prevent OOM)
            # Limit to max 2x input size to avoid memory explosion
            if stage_idx < self.n_stages - 1:  # Don't upsample after last stage
                kinet_current_size = kinet_features.shape[2:]
                max_allowed_size = tuple(s * 2 for s in input_spatial_size)
                # Only upsample if not exceeding limit
                can_upsample = all(curr < max_size for curr, max_size in zip(kinet_current_size, max_allowed_size))
                if can_upsample:
                    kinet_size = tuple(min(s * 2, max_s) for s, max_s in zip(kinet_current_size, max_allowed_size))
                    kinet_features = F.interpolate(
                        kinet_features,
                        size=kinet_size,
                        mode=self.interpolation_mode,
                        align_corners=False if self.interpolation_mode != 'nearest' else None
                    )

            # Store skip connections BEFORE CRFB (matches original KiU-Net)
            # Original stores pre-refinement features for decoder
            unet_skips.append(unet_features)
            kinet_skips.append(kinet_features)

            # Cross-Refinement Block
            unet_features, kinet_features = self.crfb_encoder[stage_idx](unet_features, kinet_features)

        # === DECODER ===
        unet_dec = unet_features  # Start from bottleneck
        kinet_dec = kinet_features

        deep_supervision_outputs = []

        # U-Net decoder
        for dec_idx in range(len(self.unet_decoder)):
            # Upsample and concatenate with skip
            unet_dec = self.unet_transpconvs[dec_idx](unet_dec)
            skip_idx = self.n_stages - 2 - dec_idx
            unet_dec = torch.cat([unet_dec, unet_skips[skip_idx]], dim=1)
            unet_dec = self.unet_decoder[dec_idx](unet_dec)

            # Deep supervision
            # Only collect from intermediate decoder stages (we created heads for these)
            if self.deep_supervision and dec_idx < len(self.unet_deep_supervision_heads):
                ds_out = self.unet_deep_supervision_heads[dec_idx](unet_dec)
                deep_supervision_outputs.append(ds_out)

        # Ki-Net decoder (goes from bottleneck back through encoder stages in reverse)
        for dec_idx in range(len(self.kinet_decoder)):
            # Downsample and concatenate with skip
            kinet_dec = self.kinet_downsample[dec_idx](kinet_dec)
            # Skip index: decoder goes from n_stages-1 -> 0, so skip is at n_stages-2-dec_idx
            skip_idx = self.n_stages - 2 - dec_idx
            skip = kinet_skips[skip_idx]

            # Match spatial dimensions if needed (due to limited upsampling in encoder)
            if kinet_dec.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(
                    skip,
                    size=kinet_dec.shape[2:],
                    mode=self.interpolation_mode,
                    align_corners=False if self.interpolation_mode != 'nearest' else None
                )

            kinet_dec = torch.cat([kinet_dec, skip], dim=1)
            kinet_dec = self.kinet_decoder[dec_idx](kinet_dec)

        # Final segmentation outputs from both branches
        unet_out = self.unet_seg_head(unet_dec)
        kinet_out = self.kinet_seg_head(kinet_dec)

        # Resize kinet output to match unet output if needed
        if kinet_out.shape[2:] != unet_out.shape[2:]:
            kinet_out = F.interpolate(
                kinet_out,
                size=unet_out.shape[2:],
                mode=self.interpolation_mode,
                align_corners=False if self.interpolation_mode != 'nearest' else None
            )

        # Fuse both branches
        final_output = (unet_out + kinet_out) / 2.0

        # Return outputs
        if self.deep_supervision:
            # Return list: [final_output, ds_out_1, ds_out_2, ...]
            return [final_output] + deep_supervision_outputs[::-1]
        else:
            return final_output
