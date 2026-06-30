
from .modules import Flatten, Activation

from typing import Optional, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialSegmentationHead(nn.Module):
    """
    Evidential head for segmentation:
      - predicts K-channel non-negative evidence via softplus
      - converts evidence -> Dirichlet parameters alpha = evidence + 1
      - returns predictive mean p_hat = alpha / sum(alpha)

    Args:
        in_channels (int): decoder feature channels.
        num_classes (int): K classes.
        kernel_size (int): conv kernel size (default 3).
        upsampling (int): bilinear upsampling factor (default 1).
        cap_evidence (Optional[float]): if set, clamps softplus output to avoid huge S.

    Forward returns:
        dict with:
            "alpha":  [B, K, H, W]  Dirichlet parameters (>=1)
            "p_hat":  [B, K, H, W]  predictive mean (used for argmax masks)
            "S":      [B, 1, H, W]  total evidence mass S = sum_k alpha_k
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        kernel_size: int = 3,
        upsampling: int = 1,
        cap_evidence: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(
            in_channels, num_classes, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.cap_evidence = cap_evidence

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e = F.softplus(self.up(self.conv(x)))  # evidence e_k >= 0
        if self.cap_evidence is not None:
            e = torch.clamp(e, max=self.cap_evidence)
        alpha = e + 1.0                         # Dirichlet parameters (>=1)
        S = alpha.sum(dim=1, keepdim=True)      # total evidence
        p_hat = alpha / S                       # predictive categorical mean
        return {"alpha": alpha, "p_hat": p_hat, "S": S}
    
class CountHead(nn.Sequential):
    """
    A head for predicting a per-pixel count or density map, similar in structure to a segmentation head.
    It consists of a single Conv2D layer with optional upsampling and an activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (e.g., for a density or count map).
        kernel_size (int): Size of the convolutional kernel. Defaults to 3.
        activation (Optional[Union[str, callable]]): Activation function name or callable.
            Can be None for no activation, or a string like 'relu', 'sigmoid', etc.
        upsampling (int): Upsampling factor. If > 1, uses nn.UpsamplingBilinear2d with the given
            scale factor. Defaults to 1 (no upsampling).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 1
    ) -> None:
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        up = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        act = Activation(activation)

        super().__init__(conv2d, up, act)


class EvidentialCentroidHeadBeta(nn.Module):
    """
    Evidential head for centroid/peak map using a 2-class Dirichlet (i.e., Beta):
      - channel 0: "centroid"   evidence
      - channel 1: "background" evidence

    Args:
        in_channels (int): decoder feature channels.
        kernel_size (int): conv kernel size (default 3).
        upsampling (int): bilinear upsampling factor (default 1).
        cap_evidence (Optional[float]): clamp on evidence to stabilize training.
        centroid_channel_first (bool): if True, channel 0 is centroid (recommended).
        return_logits_like (bool): if True, also returns a 'p_cent_vis' optionally
            passed through an Activation for quick visualization (kept off by default).
        activation (Optional[Union[str, callable]]): optional activation applied ONLY
            to a visualization-friendly map 'p_cent_vis'. NOT used for training.

    Forward returns:
        dict with:
            "alpha":   [B, 2, H, W]  Dirichlet parameters (>=1)
            "p_cent":  [B, 1, H, W]  predictive mean of the 'centroid' class
            "S":       [B, 1, H, W]  total evidence mass S = alpha_1 + alpha_2
            ("p_cent_vis": [B,1,H,W]) if return_logits_like=True and activation provided
    """
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        upsampling: int = 1,
        cap_evidence: Optional[float] = None,
        centroid_channel_first: bool = True,
        return_logits_like: bool = False,
        activation: Optional[Union[str, callable]] = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 2, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        self.cap_evidence = cap_evidence
        self.centroid_idx = 0 if centroid_channel_first else 1
        self.return_logits_like = return_logits_like
        self.vis_activation = Activation(activation) if activation is not None else None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e = F.softplus(self.up(self.conv(x)))  # [B,2,H,W]
        if self.cap_evidence is not None:
            e = torch.clamp(e, max=self.cap_evidence)
        alpha = e + 1.0                         # [B,2,H,W]
        S = alpha.sum(dim=1, keepdim=True)      # [B,1,H,W]
        p = alpha / S                           # [B,2,H,W]
        p_cent = p[:, self.centroid_idx:self.centroid_idx+1, :, :]  # [B,1,H,W]

        out = {"alpha": alpha, "p_cent": p_cent, "S": S}
        if self.return_logits_like and self.vis_activation is not None:
            out["p_cent_vis"] = self.vis_activation(p_cent)
        return out

class EvidentialCentroidHeadNIG(nn.Module):
    """
    Evidential regression head for centroid Gaussian maps using the
    Normal–Inverse-Gamma (NIG) distribution.

    Outputs four fields per pixel:
       - gamma : mean of the Gaussian map
       - nu    : evidence for the mean       ( > 0 )
       - alpha : shape parameter for variance ( > 1 )
       - beta  : scale parameter for variance ( > 0 )

    And also:
       - y_hat       : predicted mean (same as gamma)
       - sigma2_ale  : aleatoric variance  E[σ²]
       - sigma2_epi  : epistemic variance  Var[μ]
       - S           : total evidence Φ = 2ν + α

    Arguments
    ---------
    in_channels : int
        Channels of decoder input.
    kernel_size : int
        Conv kernel size (default 3).
    upsampling : int
        Bilinear upsampling factor (1 = no upsampling).
    cap_value : float or None
        Optional clamp on ν, α, β to avoid numerical issues.
    return_vis : bool
        If True, also output a visualization-friendly y_hat_vis
        passed through an activation.
    activation : str or callable, optional
        Activation for visualization only (NOT used in training).
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        upsampling: int = 1,
        cap_value: Optional[float] = None,
        return_vis: bool = False,
        activation: Optional[Union[str, callable]] = None,
    ):
        super().__init__()

        # 4 outputs: gamma, nu, alpha, beta
        self.conv = nn.Conv2d(
            in_channels, 4, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1 else nn.Identity()
        )

        self.cap_value = cap_value
        self.return_vis = return_vis
        self.vis_activation = (
            _Activation(activation) if activation is not None else None
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.up(self.conv(x))         # [B,4,H,W]
        gamma_raw, nu_raw, alpha_raw, beta_raw = torch.split(out, 1, dim=1)

        # Softplus to enforce constraints
        gamma = torch.sigmoid(gamma_raw)                   # mean can be unconstrained
        nu    = F.softplus(nu_raw)    + 1e-6   # > 0
        alpha = F.softplus(alpha_raw) + 1.0  + 1e-6  # > 1
        beta  = F.softplus(beta_raw)   + 1e-6   # > 0

        if self.cap_value is not None:
            nu    = torch.clamp(nu,    max=self.cap_value)
            alpha = torch.clamp(alpha, max=self.cap_value)
            beta  = torch.clamp(beta,  max=self.cap_value)

        # NIG moments
        # -----------
        # prediction (mean)
        y_hat = gamma

        # aleatoric σ² = β/(α - 1)
        sigma2_ale = beta / (alpha - 1.0)

        # epistemic σ² = β/(ν(α - 1))
        sigma2_epi = beta / (nu * (alpha - 1.0))

        # total evidence Φ = 2ν + α
        S = 2.0 * nu + alpha

        out_dict = {
            "gamma":       gamma,
            "nu":          nu,
            "alpha":       alpha,
            "beta":        beta,
            "y_hat":       y_hat,
            "sigma2_ale":  sigma2_ale,
            "sigma2_epi":  sigma2_epi,
            "S":           S,
        }

        # Optional visualization map
        if self.return_vis and self.vis_activation is not None:
            out_dict["y_hat_vis"] = self.vis_activation(y_hat)

        return out_dict


# Optional activation utility to mirror your Beta-head structure
class _Activation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        if isinstance(activation, str):
            if activation.lower() == "sigmoid":
                self.act = nn.Sigmoid()
            elif activation.lower() == "tanh":
                self.act = nn.Tanh()
            elif activation.lower() == "relu":
                self.act = nn.ReLU()
            else:
                raise ValueError(f"Unknown activation: {activation}")
        else:
            # assume callable
            self.act = activation

    def forward(self, x):
        return self.act(x)


class SegmentationHead(nn.Sequential):
    """
    A standard segmentation head that consists of a single Conv2D layer with optional upsampling
    and an activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (e.g., number of segmentation classes).
        kernel_size (int): Size of the convolutional kernel. Defaults to 3.
        activation (Optional[Union[str, callable]]): Activation function name or callable.
            Can be None for no activation, or a string like 'relu', 'sigmoid', etc.
        upsampling (int): Upsampling factor. If > 1, uses nn.UpsamplingBilinear2d with the given
            scale factor. Defaults to 1 (no upsampling).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 1
    ) -> None:
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        up = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        act = Activation(activation)

        super().__init__(conv2d, up, act)


class CountHead(nn.Sequential):
    """
    A head for predicting a per-pixel count or density map, similar in structure to a segmentation head.
    It consists of a single Conv2D layer with optional upsampling and an activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (e.g., for a density or count map).
        kernel_size (int): Size of the convolutional kernel. Defaults to 3.
        activation (Optional[Union[str, callable]]): Activation function name or callable.
            Can be None for no activation, or a string like 'relu', 'sigmoid', etc.
        upsampling (int): Upsampling factor. If > 1, uses nn.UpsamplingBilinear2d with the given
            scale factor. Defaults to 1 (no upsampling).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: Optional[Union[str, callable]] = None,
        upsampling: int = 1
    ) -> None:
        conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        up = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        act = Activation(activation)

        super().__init__(conv2d, up, act)


class ClassificationHead(nn.Sequential):
    """
    A classification head typically used at the end of an encoder or a feature extractor.
    It pools the spatial dimensions, optionally applies dropout, and then uses a Linear layer.

    Args:
        in_channels (int): Number of input channels (the dimension of the features).
        classes (int): Number of output classes.
        pooling (str): Either 'avg' or 'max' for Adaptive pooling across spatial dimensions.
        dropout (float): Probability of dropout. If 0, no dropout is applied.
        activation (Optional[Union[str, callable]]): Activation after the linear layer,
            e.g., 'softmax', 'sigmoid', etc. If None, no activation is applied.
    """

    def __init__(
        self,
        in_channels: int,
        classes: int,
        pooling: str = "avg",
        dropout: float = 0.2,
        activation: Optional[Union[str, callable]] = None
    ) -> None:
        if pooling not in ("max", "avg"):
            raise ValueError(f"Pooling should be one of ('max', 'avg'), got {pooling}.")
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)

        flatten = Flatten()
        drop = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        act = Activation(activation)

        super().__init__(pool, flatten, drop, linear, act)
