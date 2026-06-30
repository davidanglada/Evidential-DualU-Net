from typing import Any, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from . import initialization as init

# import your evidential heads (place them wherever you defined them)
# from ..base import EvidentialSegmentationHead, EvidentialCentroidHeadBeta
# If you kept the original heads in this file, adjust the import path accordingly.


class SegmentationModel(nn.Module):
    """
    Base segmentation model with encoder/decoder and heads.
    Now standardized to return evidential outputs as dicts.
    """

    def __init__(self) -> None:
        super().__init__()
        # self.encoder = ...
        # self.decoder = ...
        # self.segmentation_head = ...
        # self.count_head = ...
        # self.decoder_count = ...
        # self.classification_head = ...
        pass

    def initialize(self) -> None:
        """Initialize decoder and heads."""
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if hasattr(self, "count_head") and self.count_head is not None:
            init.initialize_head(self.count_head)
        if hasattr(self, "classification_head") and self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            dict with keys:
              - "seg":  {"alpha":[B,K,H,W], "p_hat":[B,K,H,W], "S":[B,1,H,W]}
              - "cent": {"alpha":[B,2,H,W], "p_cent":[B,1,H,W], "S":[B,1,H,W]}   (if count_head exists)
              - "cls":  logits or post-activated scores (if classification_head exists)
        """
        # 1) Encode
        features = self.encoder(x)

        # 2) Decode segmentation
        dec_s = self.decoder(*features)
        seg_out: Dict[str, torch.Tensor] = self.segmentation_head(dec_s)  # evidential dict

        out = {"seg": seg_out}

        # 3) Optional centroid/count head
        if hasattr(self, 'count_head') and self.count_head is not None:
            dec_c = self.decoder_count(*features)
            cent_out: Dict[str, torch.Tensor] = self.count_head(dec_c)  # evidential dict
            out["cent"] = cent_out
            out["x_cent"] = dec_c  # optional: raw features before head

        # 4) Optional classification head
        if hasattr(self, 'classification_head') and self.classification_head is not None:
            out["cls"] = self.classification_head(features[-1])

        return out

    def predict(self, x: torch.Tensor):
        """Eval wrapper that returns the same dict structure as forward()."""
        training_mode = self.training
        if training_mode:
            self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
        if training_mode:
            self.train()
        return outputs