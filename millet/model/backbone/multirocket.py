import torch
from torch import nn
import torch.nn.functional as F

from millet.model.backbone.common import manual_pad


class MultiRocketFeatureExtractor(nn.Module):
    """MultiRocket feature extractor for MILLET."""

    def __init__(self, n_in_channels: int, out_channels: int = 32, padding_mode: str = "replicate"):
        super().__init__()
        self.n_in_channels = n_in_channels
        # Define a custom encoder based on MultiRocket features
        self.instance_encoder = nn.Sequential(
            MultiRocketBlock(n_in_channels, out_channels, padding_mode),
            MultiRocketBlock(out_channels * 4, out_channels, padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_len = 21  # Specific minimum length padding
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)


class MultiRocketBlock(nn.Module):
    """A MultiRocket-style block with multiple convolutions and pooling."""

    def __init__(self, in_channels: int, out_channels: int, padding_mode: str = "replicate"):
        super().__init__()
        # Use parallel convolutions with different kernel sizes, as in MultiRocket
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=ks, padding="same", padding_mode=padding_mode)
            for ks in [3, 5, 7, 9]
        ])
        self.batch_norm = nn.BatchNorm1d(out_channels * 4)
        self.pooling_layer = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parallel convolution layers
        conv_outputs = [conv(x) for conv in self.conv_layers]
        z = torch.cat(conv_outputs, dim=1)  # Concatenate outputs
        z = self.batch_norm(z)
        z = self.pooling_layer(z)
        return F.relu(z)