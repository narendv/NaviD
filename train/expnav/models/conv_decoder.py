"""Simple convolutional decoder that maps EfficientNet-style embeddings to RGB images.

The decoder first linearly projects a latent vector into a small spatial grid and
then upsamples through a stack of transposed convolutions until reaching a
96x96 resolution. A final 1x1 convolution forms the RGB head.
"""

from typing import Dict

import torch
from torch import nn


class DeconvBlock(nn.Module):
    """Upsample by a factor of 2 using a transposed convolution followed by normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvDecoder(nn.Module):
    """Decode a 1280-D latent vector into a 96x96 RGB image."""

    def __init__(
        self,
        latent_n_channels: int = 1280,
        base_channels: int = 256,
        initial_spatial_size: int = 6,
        output_key: str = "rgb_2",
    ) -> None:
        super().__init__()
        self.latent_n_channels = latent_n_channels
        self.base_channels = base_channels
        self.initial_spatial_size = initial_spatial_size
        self.output_key = output_key

        flattened_size = base_channels * initial_spatial_size * initial_spatial_size
        self.project = nn.Sequential(
            nn.Linear(latent_n_channels, flattened_size),
            nn.ReLU(inplace=True),
        )

        # Sequence of transposed convolutions to reach 96x96.
        # 6 -> 12 -> 24 -> 48 -> 96
        self.deconv_layers = nn.ModuleList(
            [
                DeconvBlock(base_channels, base_channels),
                DeconvBlock(base_channels, base_channels // 2),
                DeconvBlock(base_channels // 2, base_channels // 4),
                DeconvBlock(base_channels // 4, max(base_channels // 8, 32)),
            ]
        )

        self.output_conv = nn.Conv2d(max(base_channels // 8, 32), 3, kernel_size=1)

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = latent.shape[0]
        x = self.project(latent)
        x = x.view(batch_size, self.base_channels, self.initial_spatial_size, self.initial_spatial_size)

        for layer in self.deconv_layers:
            x = layer(x)

        rgb = self.output_conv(x)
        return {self.output_key: rgb}


if __name__ == "__main__":
    from torchinfo import summary

    model = ConvDecoder()
    w = torch.randn((2, 1280))
    out = model(w)
    for k, v in out.items():
        print(f"{k}: {v.shape}")

    summary(model, input_size=(2, 1280))
