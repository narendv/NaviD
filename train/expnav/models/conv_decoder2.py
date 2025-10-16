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

class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm, and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
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
        output_key: str = "rgb_2",
    ) -> None:
        super().__init__()
        self.latent_n_channels = latent_n_channels
        self.base_channels = base_channels
        self.output_key = output_key

        # self.skip_map = {
        #     "reduction_4": 112,     # 6,6
        #     "reduction_3": 40,      # 12,12
        #     "reduction_2": 24,      # 24,24
        #     "reduction_1": 16,      # 48,48
        # }
        # Sequence of transposed convolutions to reach 96x96.
        # 6 -> 12 -> 24 -> 48 -> 96
        self.deconv_layers = nn.ModuleList(
            [
                ConvBlock(latent_n_channels, base_channels),
                DeconvBlock(base_channels, base_channels // 2),
                ConvBlock(base_channels // 2, base_channels // 2),
                DeconvBlock(base_channels // 2, base_channels // 4),
                ConvBlock(base_channels // 4, base_channels // 4),
                DeconvBlock(base_channels // 4, base_channels // 8),
                ConvBlock(base_channels // 8, base_channels // 8),
                DeconvBlock(base_channels // 8, max(base_channels // 16, 32)),
            ]
        )


        self.output_conv = nn.Conv2d(max(base_channels // 16, 32), 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        for layer in self.deconv_layers:
            x = layer(x)

        rgb = self.output_conv(x)
        return {self.output_key: rgb}


if __name__ == "__main__":
    from torchinfo import summary

    model = ConvDecoder(
        latent_n_channels=112,
        base_channels=512,
    )
    w = torch.randn((2, 112, 6, 6))
    out = model(w)
    for k, v in out.items():
        print(f"{k}: {v.shape}")

    summary(model, input_size=(2, 112, 6, 6))
