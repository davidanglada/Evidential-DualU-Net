# ------------------------------------------------------------------------
# Based on "Segmentation Models PyTorch": https://pypi.org/project/segmentation-models-pytorch/
# Referencing the original U-Net paper (Ronneberger et al., 2015).
# Licensed under the MIT License. See LICENSE for details.
# ------------------------------------------------------------------------
# Modifications for Monte Carlo Dropout (MCD) in DualU-Net / Multi-task U-Net architectures.
# ------------------------------------------------------------------------

from typing import Optional, List

import torch.nn as nn
from .decoder_unet_mcd import MCDUnetDecoder  # Using the MCD version of the decoder

from ..encoders import get_encoder
from ..base import SegmentationModel, SegmentationHead, CountHead


class MCDDualUNet(SegmentationModel):
    """
    MCDDualUNet is a fully convolutional neural network for cell image segmentation
    and count (density) estimation, incorporating Monte Carlo Dropout (MCD) for uncertainty estimation.

    1) **Segmentation Decoder**: Produces a semantic segmentation mask.
    2) **Count Decoder**: Produces a count/density map (e.g., for cells or nuclei).

    By default:
      - The encoder is "resnext50_32x4d" with ImageNet weights (if available).
      - Both segmentation and count decoders use `MCDUnetDecoder` to incorporate dropout.
      - No activation functions are applied in the heads (raw logits output).

    Args:
        encoder_name (str): Name of the backbone encoder, default "resnext50_32x4d".
        encoder_depth (int): Number of downsampling stages in the encoder [3..5], default 5.
        encoder_weights (Optional[str]): Pretrained weights for the encoder (e.g. "imagenet") or None.
        decoder_use_batchnorm (bool): Use BatchNorm2d in decoder blocks if True (or "inplace" for InplaceABN).
        decoder_channels (List[int]): Number of channels in each decoder stage. Must match `encoder_depth`.
        in_channels (int): Number of input channels, default 3 for RGB.
        classes_s (int): Number of output segmentation channels.
        classes_c (int): Number of output count/density channels.
        dropout_p (float): Dropout probability for MCD (default 0.2).
        aux_params (Optional[dict]): If provided, a classification head is built on top of the encoder.

    Returns:
        nn.Module: A PyTorch model implementing a dual-decoder U-Net design (segmentation + count).
    """

    def __init__(
        self,
        encoder_name: str = "resnext50_32x4d",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes_s: int = 1,
        classes_c: int = 1,
        dropout_p: float = 0.2,  # Dropout probability for MCD
        aux_params: Optional[dict] = None,
    ):
        """
        Initialize the MCDDualUNet model, creating two separate `MCDUnetDecoder` instances:
        one for segmentation, and one for counting.

        Args:
            encoder_name: Default "resnext50_32x4d".
            encoder_depth: Default 5, controlling number of downsampling stages.
            encoder_weights: Pretrained weights or None.
            decoder_use_batchnorm: Whether to use batch normalization (True/False/"inplace") in decoders.
            decoder_channels: List defining the channels in each decoder stage.
            in_channels: Number of input channels (e.g., 3 for RGB).
            classes_s: Number of segmentation output channels.
            classes_c: Number of counting/density output channels.
            dropout_p: Dropout probability for Monte Carlo Dropout (MCD).
            aux_params: Dictionary for an optional classification head (e.g. {"classes": 2, ...}).
        """
        super().__init__()

        # -----------------
        # 1) Encoder
        # -----------------
        self.encoder = get_encoder(
            encoder_name=encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # -----------------
        # 2) Segmentation Decoder & Head (with MCD)
        # -----------------
        self.decoder = MCDUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            dropout_p=dropout_p,  # MCD Dropout applied in decoder
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes_s,
            activation=None,  # No activation (raw logits)
            kernel_size=3,
        )

        # -----------------
        # 3) Count Decoder & Head (with MCD)
        # -----------------
        self.decoder_count = MCDUnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            dropout_p=dropout_p,  # MCD Dropout applied in decoder
        )
        self.count_head = CountHead(
            in_channels=decoder_channels[-1],
            out_channels=classes_c,
            activation=None,  # No activation (raw logits)
            kernel_size=3,
        )

        self.classification_head = None

        self.name = f"mcdu-dualunet-{encoder_name}"
        self.initialize()  # Provided by the base SegmentationModel


def enable_mcd(model):
    """
    Enables Monte Carlo Dropout (MCD) during inference.

    Args:
        model (nn.Module): The MCDDualUNet model.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()  # Keep dropout active during inference


def mc_dropout_forward(model, x, num_samples=30):
    """
    Performs multiple stochastic forward passes with Monte Carlo Dropout.

    Args:
        model (nn.Module): The MCDDualUNet model.
        x (torch.Tensor): Input image tensor.
        num_samples (int): Number of forward passes.

    Returns:
        mean_output (torch.Tensor): Mean prediction.
        var_output (torch.Tensor): Variance for uncertainty estimation.
    """
    enable_mcd(model)  # Ensure dropout is active

    outputs = torch.stack([model(x) for _ in range(num_samples)], dim=0)
    mean_output = outputs.mean(dim=0)  # Mean prediction
    var_output = outputs.var(dim=0)  # Uncertainty estimation

    return mean_output, var_output
