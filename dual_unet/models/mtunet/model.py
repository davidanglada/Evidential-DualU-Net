from typing import Optional, Union, List, Callable
import torch
import torch.nn as nn

from .decoder_segment import UnetDecoder_Segment
from .decoder_count import UnetDecoder_Count
from ..encoders import get_encoder
from ..base import SegmentationModel
# swap these imports to your actual module where you placed the evidential heads:
from ..base import EvidentialSegmentationHead, CountHead
from ..base import ClassificationHead


class DualUNet(SegmentationModel):
    """
    DualUNet with evidential heads:
      - Seg head: K-class Dirichlet (alpha, p_hat, S)
      - Cent head: Beta/Dirichlet-2 (alpha, p_cent, S)
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes_s: int = 6,   # ← set to 6 for your case
        classes_c: int = 2,   # ← Beta head needs 2 channels (centroid vs background)
        activation_s: Optional[Union[str, Callable]] = None,  # keep None (we use p_hat from Dirichlet)
        activation_c: Optional[Union[str, Callable]] = None,  # keep None (we use p_cent from Dirichlet-2)
        aux_params: Optional[dict] = None,
        cap_evidence_seg: Optional[float] = None,   # optional: clamp evidence for stability
        cap_evidence_cent: Optional[float] = None,  # optional: clamp evidence for stability
        upsample_s: int = 1,
        upsample_c: int = 1,
    ):
        super().__init__()

        # 1) Encoder
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # 2) Segmentation decoder + evidential head (K=classes_s)
        self.decoder = UnetDecoder_Segment(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = EvidentialSegmentationHead(
            in_channels=decoder_channels[-1],
            num_classes=classes_s,
            kernel_size=3,
            upsampling=upsample_s,
            cap_evidence=cap_evidence_seg,
        )

        # 3) Count/centroid decoder + evidential Beta head (2 classes)
        self.decoder_count = UnetDecoder_Count(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        self.count_head = CountHead(
            in_channels=decoder_channels[-1],
            out_channels=classes_c,
            activation=activation_c,
            kernel_size=3,
        )

        # 4) Optional classification head (unchanged)
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                **aux_params
            )
        else:
            self.classification_head = None

        self.name = f"dual_unet_{encoder_name}"
        self.initialize()