import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalTemperatureScaling(nn.Module):
    """
    A tree-like convolutional network for Local Temperature Scaling (LTS).
    It follows the schematic from the figure in the supplementary:
      - Leaf nodes (v1..v5) are small conv filters
      - Internal nodes (c5..c8) produce gating values (sigmoid) that mix their children
      - The image-based path (v5) merges at the top node with the logits-based path
      - The final output is a spatially varying temperature map T(x) > 0

    Usage:
        net = TreeLikeConvNetLTS(num_logit_channels=20, image_channels=3, hidden_channels=8)
        temperature_map = net(logits, image)
        # logits: (B, num_logit_channels, H, W)
        # image:  (B, image_channels,   H, W)
    """
    def __init__(self,
                 num_logit_channels: int,
                 image_channels: int = 3,
                 hidden_channels: int = 8,
                 eps: float = 1e-4):
        """
        Args:
            num_logit_channels: number of channels in the logits input
            image_channels: number of channels in the image input
            hidden_channels: number of intermediate channels for gating convolutions
            eps: a small constant added at the final step to ensure positivity
        """
        super().__init__()
        self.eps = eps
        
        # --- Leaf node convolutions (v1..v4 for logits, v5 for image) ---
        # All use kernel_size=5, dilation=2 => effective receptive field of 9x9
        # We omit biases for clarity, but you can add them if desired.
        self.v1 = nn.Conv2d(num_logit_channels, 1, kernel_size=5, dilation=2, padding=4, bias=True)
        self.v2 = nn.Conv2d(num_logit_channels, 1, kernel_size=5, dilation=2, padding=4, bias=True)
        self.v3 = nn.Conv2d(num_logit_channels, 1, kernel_size=5, dilation=2, padding=4, bias=True)
        self.v4 = nn.Conv2d(num_logit_channels, 1, kernel_size=5, dilation=2, padding=4, bias=True)
        self.v5 = nn.Conv2d(image_channels,      1, kernel_size=5, dilation=2, padding=4, bias=True)
        
        # --- Gating (internal) node convolutions (c5..c8) ---
        # Each gating conv produces a single channel that is passed through a sigmoid
        self.c5 = nn.Conv2d(num_logit_channels, 1, kernel_size=5, dilation=2, padding=4, bias=True)
        self.c6 = nn.Conv2d(num_logit_channels, 1, kernel_size=5, dilation=2, padding=4, bias=True)
        self.c7 = nn.Conv2d(num_logit_channels, 1, kernel_size=5, dilation=2, padding=4, bias=True)
        self.c8 = nn.Conv2d(num_logit_channels, 1, kernel_size=5, dilation=2, padding=4, bias=True)

    def forward(self, logits: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to produce a local temperature map T(x). 
        The final temperature is ReLU( top_node ) + eps to ensure positivity.
        
        Args:
            logits: (B, num_logit_channels, H, W)
            image:  (B, image_channels, H, W)
        Returns:
            temperature: (B, 1, H, W), a spatial map of temperature > 0
        """
        # Leaf nodes for logits
        v1_out = self.v1(logits)  # (B,1,H,W)
        v2_out = self.v2(logits)  # ...
        v3_out = self.v3(logits)
        v4_out = self.v4(logits)
        
        # Leaf node for image
        v5_out = self.v5(image)   # (B,1,H,W)
        
        # Gating internal nodes produce sigmoids
        # c5_out = sigma(c5 * logits)
        c5_out = torch.sigmoid(self.c5(logits))  # (B,1,H,W)
        c6_out = torch.sigmoid(self.c6(logits))
        c7_out = torch.sigmoid(self.c7(logits))
        c8_out = torch.sigmoid(self.c8(logits))
        
        # In the figure, c5/c6/c7 are used to mix among v1..v4 paths
        # For example (pseudocode):
        #   node_left = c5_out * ( c6_out * v1_out + (1 - c6_out)*v2_out )
        #   node_right= (1-c5_out)* ( c7_out * v3_out + (1 - c7_out)*v4_out )
        #   logits_branch = node_left + node_right
        # Then we combine the logits_branch with the image_branch via c8_out.
        
        # Example of mixing:
        node_a = c6_out * v1_out + (1.0 - c6_out) * v2_out  # mix v1, v2
        node_b = c7_out * v3_out + (1.0 - c7_out) * v4_out  # mix v3, v4
        logits_branch = c5_out * node_a + (1.0 - c5_out) * node_b
        
        # Now combine logits_branch with image_branch v5_out using c8_out
        # top_node = ReLU( c8_out * logits_branch + (1 - c8_out) * v5_out ) + eps
        # We'll do a ReLU for positivity, then add self.eps
        top_node = c8_out * logits_branch + (1.0 - c8_out) * v5_out
        
        # Finally, ensure positivity
        temperature_map = F.relu(top_node) + self.eps

        calibrated_logits = logits / temperature_map
        
        return calibrated_logits, temperature_map