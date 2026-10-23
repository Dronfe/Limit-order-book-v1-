"""Dual-stream CNN with cross-attention fusion for lunar ice mapping.

Stream A encodes OHRC optical surface morphology (illumination, roughness,
crater rims). Stream B encodes DFSAR polarimetric backscatter (HH, HV, VH,
VV), which is sensitive to subsurface dielectric contrasts associated with
water ice. A bidirectional cross-attention module mixes the two embeddings
and a lightweight decoder produces a dense ice-probability map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with batch-norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    """Stride-2 downsample followed by a residual-style conv block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class OpticalStream(nn.Module):
    """Encoder for OHRC optical surface features (Stream A)."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.stem = ConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.out_channels = base_channels * 8

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        return s4, [s1, s2, s3]


class RadarStream(nn.Module):
    """Encoder for DFSAR polarimetric backscatter (Stream B).

    Four input channels correspond to the HH, HV, VH, and VV DFSAR products.
    A 1x1 mixing layer learns polarimetric combinations before spatial
    encoding.
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        self.polarimetric_mix = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.stem = ConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8)
        self.out_channels = base_channels * 8

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.polarimetric_mix(x)
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        return s4, [s1, s2, s3]


class CrossAttentionFusion(nn.Module):
    """Bidirectional spatial cross-attention between optical and radar streams.

    Optical tokens query radar keys/values (surface context attending into
    subsurface scatterers) and vice versa. The two attended maps are fused
    with a 1x1 convolution.
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_opt = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_rad = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_rad = nn.Conv2d(channels, channels, kernel_size=1)

        self.q_rad = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_opt = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_opt = nn.Conv2d(channels, channels, kernel_size=1)

        self.proj_opt = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_rad = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.norm_opt = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.norm_rad = nn.GroupNorm(num_groups=8, num_channels=channels)

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-head attention over flattened spatial tokens.

        Args:
            query, key, value: Tensors of shape ``(B, C, H, W)``.

        Returns:
            Attended tensor of shape ``(B, C, H, W)``.
        """
        b, c, h, w = query.shape
        q = query.view(b, self.num_heads, self.head_dim, h * w).transpose(-2, -1)
        k = key.view(b, self.num_heads, self.head_dim, h * w)
        v = value.view(b, self.num_heads, self.head_dim, h * w).transpose(-2, -1)

        attn = torch.matmul(q, k) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)
        return out

    def forward(self, optical: torch.Tensor, radar: torch.Tensor) -> torch.Tensor:
        if optical.shape != radar.shape:
            raise ValueError(
                f"Stream feature maps must match, got {tuple(optical.shape)} vs {tuple(radar.shape)}"
            )

        opt = self.norm_opt(optical)
        rad = self.norm_rad(radar)

        # Optical queries radar (surface morphology conditions subsurface cues).
        opt_attended = self.proj_opt(
            self._attend(self.q_opt(opt), self.k_rad(rad), self.v_rad(rad))
        )
        # Radar queries optical (dielectric anomalies grounded in surface context).
        rad_attended = self.proj_rad(
            self._attend(self.q_rad(rad), self.k_opt(opt), self.v_opt(opt))
        )

        opt_out = optical + self.dropout(opt_attended)
        rad_out = radar + self.dropout(rad_attended)
        return self.fusion(torch.cat([opt_out, rad_out], dim=1))


class DecoderBlock(nn.Module):
    """Bilinear upsample, skip concat, and conv block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class DualStreamIceDetector(nn.Module):
    """Dual-stream CNN that outputs a spatial ice-probability map.

    Args:
        optical_channels: Number of OHRC bands (typically 1).
        radar_channels: Number of DFSAR polarisations (HH, HV, VH, VV).
        base_channels: Width of the first encoder stage.
        num_heads: Attention heads in the fusion module.
        dropout: Dropout applied inside cross-attention.

    Input:
        optical: ``(B, optical_channels, H, W)``
        radar: ``(B, radar_channels, H, W)``

    Output:
        Ice probability logits ``(B, 1, H, W)``. Apply ``sigmoid`` for
        probabilities in ``[0, 1]``.
    """

    def __init__(
        self,
        optical_channels: int = 1,
        radar_channels: int = 4,
        base_channels: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.optical_stream = OpticalStream(optical_channels, base_channels)
        self.radar_stream = RadarStream(radar_channels, base_channels)

        bottleneck = base_channels * 8
        self.fusion = CrossAttentionFusion(bottleneck, num_heads=num_heads, dropout=dropout)

        self.up1 = DecoderBlock(bottleneck, base_channels * 4, base_channels * 4)
        self.up2 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up3 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(
        self,
        optical: torch.Tensor,
        radar: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        opt_feat, opt_skips = self.optical_stream(optical)
        rad_feat, rad_skips = self.radar_stream(radar)

        fused = self.fusion(opt_feat, rad_feat)

        # Average skip connections from both streams so the decoder restores
        # both optical texture and polarimetric structure.
        skips = [(o + r) * 0.5 for o, r in zip(opt_skips, rad_skips)]
        x = self.up1(fused, skips[2])
        x = self.up2(x, skips[1])
        x = self.up3(x, skips[0])
        logits = self.head(x)

        if logits.shape[-2:] != optical.shape[-2:]:
            logits = F.interpolate(
                logits, size=optical.shape[-2:], mode="bilinear", align_corners=False
            )
        return torch.sigmoid(logits) if return_probs else logits
