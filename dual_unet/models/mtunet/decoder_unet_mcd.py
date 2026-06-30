# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for Monte Carlo Dropout (MCD) in DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md


class MCDDecoderBlock(nn.Module):
    """
    A Monte Carlo Dropout (MCD) U-Net decoder block that:
      1) Upsamples the incoming feature map by a factor of 2 (nearest-neighbor).
      2) Concatenates it with a corresponding skip connection (if not None).
      3) Applies two consecutive convolution + ReLU layers (with optional BatchNorm and Dropout).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        dropout_p: float = 0.5,  # Dropout probability for MCD
    ):
        """
        Args:
            in_channels (int): Number of input channels (before concatenation).
            skip_channels (int): Number of channels in the skip connection.
            out_channels (int): Number of output channels after the convolutions.
            use_batchnorm (bool): If True, use BatchNorm2d between Conv2D and ReLU.
            dropout_p (float): Dropout probability for Monte Carlo Dropout.
        """
        super().__init__()

        # First convolution: in_channels + skip_channels -> out_channels
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        # Dropout added after first conv
        self.dropout1 = nn.Dropout(p=dropout_p)

        # Second convolution: out_channels -> out_channels
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        # Dropout added after second conv
        self.dropout2 = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MCD decoder block.

        Args:
            x (torch.Tensor): Feature map from the previous decoder step.
            skip (torch.Tensor): Corresponding skip connection feature map (can be None).

        Returns:
            torch.Tensor: Output feature map of shape (N, out_channels, H_out, W_out).
        """
        # 1) Upsample
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        # 2) Concatenate skip connection if available
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        # 3) Two consecutive convs with dropout
        x = self.conv1(x)
        x = self.dropout1(x)  # Apply dropout after first conv
        x = self.conv2(x)
        x = self.dropout2(x)  # Apply dropout after second conv

        return x


class MCDCenterBlock(nn.Sequential):
    """
    Optional Monte Carlo Dropout (MCD) center block used at the transition between encoder and decoder stages.
    Consists of two consecutive Conv2d+ReLU layers with dropout for uncertainty estimation.
    """

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True, dropout_p: float = 0.2):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_batchnorm (bool): If True, use BatchNorm2d in each conv layer.
            dropout_p (float): Dropout probability for Monte Carlo Dropout.
        """
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        dropout1 = nn.Dropout(p=dropout_p)

        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        dropout2 = nn.Dropout(p=dropout_p)

        super().__init__(conv1, dropout1, conv2, dropout2)


class MCDUnetDecoder(nn.Module):
    """
    A Monte Carlo Dropout (MCD) U-Net decoder with dropout support, which:
      1) Optionally applies an MCDCenterBlock if `center=True`.
      2) Iteratively builds decoder blocks to combine upsampled features
         with skip connections from the encoder.
      3) Produces the final high-resolution feature map.
    """

    def __init__(
        self,
        encoder_channels: list,
        decoder_channels: list,
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        center: bool = False,
        dropout_p: float = 0.5,  # Dropout probability for MCD
    ):
        """
        Args:
            encoder_channels (list): List of channel sizes from the encoder, e.g. [512, 256, 128, 64, 32].
            decoder_channels (list): List of channel sizes in the decoder, e.g. [256,128,64,32,16].
                                     Length must match `n_blocks`.
            n_blocks (int): Number of decoder blocks. Must match the length of `decoder_channels`.
            use_batchnorm (bool): If True, use BatchNorm2d in all decoder conv layers.
            center (bool): Whether to add an MCDCenterBlock at the beginning of the decoder.
            dropout_p (float): Dropout probability for Monte Carlo Dropout.
        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but `decoder_channels` has length {len(decoder_channels)}."
            )

        # Remove first encoder skip with same resolution as input
        encoder_channels = encoder_channels[1:][::-1]  # Reverse order

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # Center block (optional) with dropout
        if center:
            self.center = MCDCenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm, dropout_p=dropout_p)
        else:
            self.center = nn.Identity()

        # Build decoder blocks with dropout
        self.blocks = nn.ModuleList([
            MCDDecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm, dropout_p)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ])

        # self.drop = nn.Dropout(p=dropout_p)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: A list of Tensors from the encoder (skips), typically in ascending
                      spatial resolution order (from deepest to shallowest).
                      The first element is the output of the deepest encoder layer.

        Returns:
            torch.Tensor: The final decoder output after upsampling all stages.
        """
        features = features[1:][::-1]  # Remove first skip and reverse order

        head, skips = features[0], features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
