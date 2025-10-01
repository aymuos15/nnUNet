"""
Dynamic UIU-Net implementation for nnU-Net framework.

Based on: "UIU-Net: U-Net in U-Net for Infrared Small Object Detection"
(https://github.com/danfenghong/IEEE_TIP_UIU-Net)

Key features:
- Nested U-Net structure (RSU blocks contain internal U-Nets)
- Multiple side outputs with deep supervision
- Uncertainty-inspired fusion of predictions
- Fully dynamic to adapt to nnU-Net's planning system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type
from torch.nn.modules.dropout import _DropoutNd
import math


def _upsample_like(x, size):
    """Upsample tensor to target size using trilinear interpolation."""
    # If size is a tensor, extract spatial dimensions
    if isinstance(size, torch.Tensor):
        size = tuple(size.shape[2:])
    return F.interpolate(x, size=size, mode='trilinear', align_corners=False)


def _size_map(x, height):
    """
    Create size map for multi-scale processing.

    Args:
        x: Input tensor
        height: Number of scales

    Returns:
        Dictionary mapping scale index to spatial size
    """
    size = list(x.shape[-3:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class SpatialAttention3D(nn.Module):
    """
    3D Spatial attention module.

    Learns spatial importance by aggregating channel information via
    max and average pooling, then applying a convolution to generate
    attention weights.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Aggregate across channels
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        # Concatenate and generate attention
        attention = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)

        return x * attention


class ChannelAttention3D(nn.Module):
    """
    3D Channel attention module.

    Learns channel importance using global context via adaptive pooling
    and fully connected layers.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Ensure reduced channels is at least 1
        reduced_channels = max(channels // reduction, 1)

        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv3d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        # Combine and generate attention
        attention = self.sigmoid(avg_out + max_out)

        return x * attention


class InteractiveCrossAttention3D(nn.Module):
    """
    Interactive Cross-Attention module for fusing multi-scale side outputs.

    Based on UIU-Net's IC-A module, this applies spatial and channel attention
    to allow interactive feature refinement before fusion.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Channel reduction
        self.reduce = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Attention modules
        self.channel_attention = ChannelAttention3D(out_channels)
        self.spatial_attention = SpatialAttention3D(kernel_size=7)

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Concatenated side outputs [B, C_total, D, H, W]

        Returns:
            Refined fused features [B, C_out, D, H, W]
        """
        # Reduce channels
        x = self.reduce(x)

        # Apply channel attention
        x = self.channel_attention(x)

        # Apply spatial attention
        x = self.spatial_attention(x)

        # Final refinement
        x = self.refine(x)

        return x


class REBNCONV3D(nn.Module):
    """
    Basic convolutional block: Conv3D → Norm → Activation.

    This is the fundamental building block used in RSU modules.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        dilate: int = 1,
        conv_op: Type[nn.Module] = nn.Conv3d,
        norm_op: Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        conv_bias: bool = True,
    ):
        super(REBNCONV3D, self).__init__()

        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}

        # Calculate padding to maintain spatial dimensions
        if isinstance(kernel_size, int):
            padding = (kernel_size // 2) * dilate
        else:
            padding = tuple((k // 2) * dilate for k in kernel_size)

        self.conv = conv_op(in_ch, out_ch, kernel_size, padding=padding, dilation=dilate, bias=conv_bias)
        self.norm = norm_op(out_ch, **norm_op_kwargs)
        self.nonlin = nonlin(**nonlin_kwargs)

    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))


class DynamicRSU3D(nn.Module):
    """
    Dynamic Residual U-block (RSU) with nested U-Net structure.

    RSU is the core building block of UIU-Net. It creates a U-Net-like structure
    within each encoder/decoder stage, enabling multi-scale feature extraction.

    Args:
        height: Depth of the internal U-Net (number of downsampling steps)
        in_ch: Input channels
        mid_ch: Intermediate channels in the internal U-Net
        out_ch: Output channels
        kernel_size: Convolution kernel size
        stride: Stride for downsampling (typically 2)
        dilated: If True, use dilated convolutions instead of downsampling
        conv_op: Convolution operation (nn.Conv2d or nn.Conv3d)
        norm_op: Normalization operation
        norm_op_kwargs: Normalization parameters
        nonlin: Nonlinearity
        nonlin_kwargs: Nonlinearity parameters
        conv_bias: Whether to use bias in convolutions
    """

    def __init__(
        self,
        height: int,
        in_ch: int,
        mid_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 2,
        dilated: bool = False,
        conv_op: Type[nn.Module] = nn.Conv3d,
        norm_op: Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        conv_bias: bool = True,
    ):
        super(DynamicRSU3D, self).__init__()

        self.height = height
        self.dilated = dilated

        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}

        # Input convolution
        self.rebnconvin = REBNCONV3D(
            in_ch, out_ch, kernel_size, 1, conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, conv_bias
        )

        # Downsampling operation
        if conv_op == nn.Conv3d:
            self.downsample = nn.MaxPool3d(stride, stride=stride, ceil_mode=True)
        else:
            self.downsample = nn.MaxPool2d(stride, stride=stride, ceil_mode=True)

        # Build internal U-Net structure
        # Encoder path
        self.rebnconv1 = REBNCONV3D(
            out_ch, mid_ch, kernel_size, 1, conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, conv_bias
        )
        self.rebnconv1d = REBNCONV3D(
            mid_ch * 2, out_ch, kernel_size, 1, conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, conv_bias
        )

        # Intermediate layers with increasing dilation if dilated=True
        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            # Encoder
            setattr(
                self,
                f'rebnconv{i}',
                REBNCONV3D(
                    mid_ch, mid_ch, kernel_size, dilate, conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, conv_bias
                )
            )
            # Decoder
            setattr(
                self,
                f'rebnconv{i}d',
                REBNCONV3D(
                    mid_ch * 2, mid_ch, kernel_size, dilate, conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, conv_bias
                )
            )

        # Bottleneck
        dilate = 2 if not dilated else 2 ** (height - 1)
        setattr(
            self,
            f'rebnconv{height}',
            REBNCONV3D(
                mid_ch, mid_ch, kernel_size, dilate, conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, conv_bias
            )
        )

    def forward(self, x):
        """
        Forward pass through RSU block.

        Creates a nested U-Net structure:
        - Downsamples features progressively
        - Concatenates encoder features with upsampled decoder features
        - Adds residual connection from input
        """
        sizes = _size_map(x, self.height)
        x_in = self.rebnconvin(x)

        def unet(x, height=1):
            """Recursive U-Net structure."""
            if height < self.height:
                # Encoder path
                x1 = getattr(self, f'rebnconv{height}')(x)

                # Downsample and recurse (if not dilated and not at bottom)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(self.downsample(x1), height + 1)
                    # Upsample x2 to match x1's spatial size before concatenation
                    x2 = _upsample_like(x2, x1)
                else:
                    x2 = unet(x1, height + 1)

                # Decoder path - concatenate and process
                x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))

                return x
            else:
                # Bottleneck
                return getattr(self, f'rebnconv{height}')(x)

        # Residual connection
        return x_in + unet(x_in)


class DynamicUIUNet3D(nn.Module):
    """
    Dynamic UIU-Net architecture for nnU-Net framework.

    UIU-Net uses nested U-Net blocks (RSU) at each stage and produces multiple
    side outputs that are fused for final prediction. This implementation is
    fully dynamic and adapts to nnU-Net's planning system.

    Args:
        input_channels: Number of input channels
        num_classes: Number of output classes
        n_stages: Number of encoder/decoder stages (from nnU-Net planning)
        features_per_stage: Feature channels at each stage (from nnU-Net planning)
        kernel_sizes: Kernel sizes per stage (from nnU-Net planning)
        strides: Strides per stage (from nnU-Net planning)
        n_conv_per_stage: Number of convolutions per stage (from nnU-Net planning)
        n_conv_per_stage_decoder: Number of convolutions per decoder stage
        conv_op: Convolution operation (nn.Conv2d or nn.Conv3d)
        norm_op: Normalization operation
        norm_op_kwargs: Normalization parameters
        dropout_op: Dropout operation
        dropout_op_kwargs: Dropout parameters
        nonlin: Nonlinearity
        nonlin_kwargs: Nonlinearity parameters
        deep_supervision: Enable deep supervision
        conv_bias: Whether to use bias in convolutions
        rsu_heights: RSU heights per stage (None = auto-calculate)
        minimal: Use minimal RSU heights for memory efficiency
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int,
        features_per_stage: Union[List[int], Tuple[int, ...]],
        kernel_sizes: Union[List[Union[int, List[int], Tuple[int, ...]]], Tuple[Union[int, List[int], Tuple[int, ...]], ...]],
        strides: Union[List[Union[int, List[int], Tuple[int, ...]]], Tuple[Union[int, List[int], Tuple[int, ...]], ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[nn.Module] = nn.Conv3d,
        norm_op: Type[nn.Module] = nn.InstanceNorm3d,
        norm_op_kwargs: dict = None,
        dropout_op: Type[_DropoutNd] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = True,
        conv_bias: bool = True,
        rsu_heights: List[int] = None,
        minimal: bool = False,
    ):
        super(DynamicUIUNet3D, self).__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.n_stages = n_stages
        self.features_per_stage = features_per_stage
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.conv_op = conv_op
        self.deep_supervision = deep_supervision

        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}

        # Determine RSU heights per stage
        if rsu_heights is None:
            # Auto-calculate: start at 7 (or 5 for minimal), decrease per stage, min 3
            base_height = 5 if minimal else 7
            rsu_heights = []
            for i in range(n_stages):
                h = max(base_height - i, 3)
                rsu_heights.append(h)

        self.rsu_heights = rsu_heights

        # Add decoder wrapper for nnU-Net compatibility
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

        # Downsampling operation
        if conv_op == nn.Conv3d:
            self.downsample = nn.MaxPool3d(2, stride=2, ceil_mode=True)
        else:
            self.downsample = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Build encoder stages (RSU blocks)
        self.encoder_stages = nn.ModuleList()
        for i in range(n_stages):
            in_ch = input_channels if i == 0 else features_per_stage[i - 1]
            out_ch = features_per_stage[i]
            mid_ch = out_ch // 2

            # Use dilated RSU for last 2 stages
            dilated = i >= n_stages - 2

            rsu = DynamicRSU3D(
                height=rsu_heights[i],
                in_ch=in_ch,
                mid_ch=mid_ch,
                out_ch=out_ch,
                kernel_size=kernel_sizes[i] if isinstance(kernel_sizes[i], int) else kernel_sizes[i][0],
                stride=strides[i] if isinstance(strides[i], int) else strides[i][0],
                dilated=dilated,
                conv_op=conv_op,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                conv_bias=conv_bias,
            )
            self.encoder_stages.append(rsu)

        # Build decoder stages (RSU blocks)
        self.decoder_stages = nn.ModuleList()
        for i in range(n_stages - 1):
            # Decoder processes concatenated features from encoder skip and upsampled lower stage
            in_ch = features_per_stage[n_stages - 1 - i] + features_per_stage[n_stages - 2 - i]
            out_ch = features_per_stage[n_stages - 2 - i]
            mid_ch = out_ch // 2

            # Use dilated RSU for first decoder stage (deepest)
            dilated = i == 0

            # RSU height mirrors encoder
            height = rsu_heights[n_stages - 2 - i]

            rsu = DynamicRSU3D(
                height=height,
                in_ch=in_ch,
                mid_ch=mid_ch,
                out_ch=out_ch,
                kernel_size=kernel_sizes[n_stages - 2 - i] if isinstance(kernel_sizes[n_stages - 2 - i], int) else kernel_sizes[n_stages - 2 - i][0],
                stride=strides[n_stages - 2 - i] if isinstance(strides[n_stages - 2 - i], int) else strides[n_stages - 2 - i][0],
                dilated=dilated,
                conv_op=conv_op,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                conv_bias=conv_bias,
            )
            self.decoder_stages.append(rsu)

        # Side output heads (for UIU-Net fusion)
        # Create heads for bottleneck + all decoder stages
        self.side_heads = nn.ModuleList()
        # Bottleneck head
        self.side_heads.append(conv_op(features_per_stage[-1], num_classes, 3, padding=1, bias=True))
        # Decoder heads
        for i in range(n_stages - 1):
            out_ch = features_per_stage[n_stages - 2 - i]
            self.side_heads.append(conv_op(out_ch, num_classes, 3, padding=1, bias=True))

        # Fusion layer (UIU-Net's innovation)
        # Interactive Cross-Attention fusion for multi-scale side outputs
        self.fusion = InteractiveCrossAttention3D(
            in_channels=n_stages * num_classes,
            out_channels=num_classes
        )

        # Final output layer
        self.output_conv = conv_op(num_classes, num_classes, 1, bias=True)

    def forward(self, x):
        """
        Forward pass through UIU-Net.

        Returns:
            If deep_supervision: [fused_output, aux_output_1, aux_output_2, ...]
            Else: fused_output
        """
        input_size = x.shape[2:]
        sizes = _size_map(x, self.n_stages)

        # Storage for encoder features (skip connections)
        encoder_features = []

        # Storage for side outputs (for fusion)
        side_outputs = []

        # Storage for deep supervision outputs
        deep_supervision_outputs = []

        # === ENCODER ===
        x_enc = x
        for i, encoder_stage in enumerate(self.encoder_stages):
            x_enc = encoder_stage(x_enc)
            encoder_features.append(x_enc)

            # Downsample for next stage (except last)
            if i < self.n_stages - 1:
                x_enc = self.downsample(x_enc)

        # Side output from bottleneck
        bottleneck_side = self.side_heads[0](encoder_features[-1])
        bottleneck_side_upsampled = _upsample_like(bottleneck_side, input_size)
        side_outputs.append(bottleneck_side_upsampled)

        # === DECODER ===
        x_dec = encoder_features[-1]
        for i, decoder_stage in enumerate(self.decoder_stages):
            # Concatenate with encoder skip connection at same resolution
            skip_idx = self.n_stages - 2 - i

            # Upsample x_dec to match encoder feature's actual spatial size
            x_dec = _upsample_like(x_dec, encoder_features[skip_idx])
            x_dec = torch.cat([x_dec, encoder_features[skip_idx]], dim=1)

            # Process through decoder RSU
            x_dec = decoder_stage(x_dec)

            # Side output head
            side_out = self.side_heads[i + 1](x_dec)
            side_out_upsampled = _upsample_like(side_out, input_size)
            side_outputs.append(side_out_upsampled)

            # Collect for deep supervision at native resolution
            # nnU-Net expects outputs at progressively lower resolutions
            if self.deep_supervision:
                deep_supervision_outputs.append(side_out)

        # === FUSION ===
        # UIU-Net's key innovation: fuse all side outputs with attention
        side_outputs.reverse()  # Order: highest res → lowest res
        fused_sides = torch.cat(side_outputs, dim=1)
        fused_features = self.fusion(fused_sides)  # Apply attention fusion
        fused_output = self.output_conv(fused_features)  # Final prediction

        # === RETURN ===
        if self.deep_supervision:
            # Hybrid approach for nnU-Net compatibility:
            # - Main output: fused prediction at full resolution
            # - Auxiliary outputs: native resolution (progressively lower)
            # Deep supervision outputs are in decoder order, reverse for nnU-Net (highest res first)
            return [fused_output] + deep_supervision_outputs[::-1]
        else:
            return fused_output
