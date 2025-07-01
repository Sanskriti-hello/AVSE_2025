"""
Swin Transformer V2 Video Encoder for Lip Reading
Adapted for Audio-Visual Speech Enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
from einops import rearrange, repeat

from ..modules.swin_modules import SwinTransformerV2Block, PatchMerging3D


class SwinV2VideoEncoder(nn.Module):
    """
    Swin Transformer V2 for video encoding with 3D spatiotemporal modeling
    Based on: https://arxiv.org/abs/2111.09883
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (88, 88),
        patch_size: Tuple[int, int] = (4, 4),
        temporal_patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        pretrained_window_sizes: List[int] = [0, 0, 0, 0],
        **kwargs
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # 3D patch embedding for video
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # Absolute position embedding
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, patches_resolution[0] * patches_resolution[1], embed_dim)
        )
        nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None,
                pretrained_window_size=pretrained_window_sizes[i_layer]
            )
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)
        
        # Temporal modeling with TCN
        self.temporal_conv = TemporalConvNet(
            num_inputs=self.num_features,
            num_channels=[512, 512, 512],
            kernel_size=3,
            dropout=dropout
        )
        
        # Output projection
        self.head = nn.Linear(512, embed_dim)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [B, T, C, H, W] video frames
            mask: [B, T] optional temporal mask
            
        Returns:
            [B, T, D] encoded video features
        """
        B, T, C, H, W = x.shape
        
        # Reshape for 3D patch embedding
        x = rearrange(x, 'b t c h w -> b c t h w')
        
        # 3D Patch embedding
        x = self.patch_embed(x)  # [B, L, D]
        
        # Add absolute position embedding
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        # Forward through Swin layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)  # [B, L, D]
        
        # Reshape back to temporal sequence
        # Assuming spatial dimensions are reduced through patch merging
        H_out = self.patches_resolution[0] // (2 ** (self.num_layers - 1))
        W_out = self.patches_resolution[1] // (2 ** (self.num_layers - 1))
        T_out = T // self.temporal_patch_size
        
        # Global average pooling over spatial dimensions
        x = rearrange(x, 'b (t h w) d -> b t (h w) d', t=T_out, h=H_out, w=W_out)
        x = x.mean(dim=2)  # [B, T_out, D]
        
        # Temporal convolution for sequence modeling
        x = x.transpose(1, 2)  # [B, D, T_out]
        x = self.temporal_conv(x)  # [B, 512, T_out]
        x = x.transpose(1, 2)  # [B, T_out, 512]
        
        # Output projection
        x = self.head(x)  # [B, T_out, embed_dim]
        
        # Interpolate to original temporal resolution if needed
        if x.size(1) != T:
            x = F.interpolate(
                x.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        return x


class PatchEmbed3D(nn.Module):
    """3D patch embedding for video"""
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (88, 88),
        patch_size: Tuple[int, int] = (4, 4),
        temporal_patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1]
        ]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # 3D convolution for spatiotemporal patch embedding
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(temporal_patch_size, patch_size[0], patch_size[1]),
            stride=(temporal_patch_size, patch_size[0], patch_size[1])
        )
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] video tensor
        Returns:
            [B, L, D] patch embeddings
        """
        B, C, T, H, W = x.shape
        
        # 3D convolution
        x = self.proj(x)  # [B, D, T', H', W']
        
        # Flatten spatiotemporal dimensions
        x = x.flatten(2).transpose(1, 2)  # [B, T'*H'*W', D]
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x


class BasicLayer(nn.Module):
    """Basic layer for Swin Transformer V2"""
    
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        drop_path: List[float] = None,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        pretrained_window_size: int = 0
    ):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerV2Block(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                drop_path=drop_path[i] if drop_path is not None else 0.0,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size
            )
            for i in range(depth)
        ])
        
        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                norm_layer=norm_layer
            )
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the basic layer"""
        for blk in self.blocks:
            x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for sequence modeling"""
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            ]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D, T] input features
        Returns:
            [B, D_out, T] processed features
        """
        return self.network(x)


class TemporalBlock(nn.Module):
    """Temporal block for TCN"""
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                n_inputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                n_outputs, n_outputs, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove extra padding from temporal convolution"""
    
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return x[:, :, :-self.chomp_size].contiguous()