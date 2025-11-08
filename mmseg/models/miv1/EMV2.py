import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InfoFusionModule(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Feature projection networks with consistent dimensions
        self.visual_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.GroupNorm(8, in_channels),
            nn.ReLU()
        )

        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Enhanced mutual information estimator
        self.mi_estimator = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def compute_channel_entropy(self, x):
        # Compute entropy along channel dimension
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        x = F.softmax(x, dim=2)
        entropy = -torch.sum(x * torch.log(x + 1e-10), dim=2)  # [B, C]
        return entropy

    def compute_spatial_entropy(self, x):
        # Compute entropy in spatial dimensions
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        x = F.softmax(x, dim=1)
        entropy = -torch.sum(x * torch.log(x + 1e-10), dim=1)  # [B, HW]
        return entropy.view(b, h, w)

    def estimate_mutual_info(self, f1, f2):
        # Enhanced mutual information estimation with spatial preservation
        b, c, h, w = f1.shape

        # Concatenate features along channel dimension
        joint_feat = torch.cat([f1, f2], dim=1)

        # Estimate mutual information while preserving spatial dimensions
        mi_weights = self.mi_estimator(joint_feat)  # [B, C, H, W]

        return mi_weights

    def forward(self, visual_feat, semantic_feat):
        """
        Args:
            visual_feat: [B, C, H, W]
            semantic_feat: [B, C, H, W]  # Assuming semantic_feat has been properly reshaped
        """
        B, C, H, W = visual_feat.shape

        # Project features
        v_proj = self.visual_proj(visual_feat)  # [B, C, H, W]
        s_proj = self.visual_proj(semantic_feat)  # [B, C, H, W]

        # 1. Channel Attention weights
        channel_entropy = self.compute_channel_entropy(v_proj)  # [B, C]
        channel_weights = self.channel_attention(channel_entropy)  # [B, C]

        # 2. Spatial Attention weights
        spatial_entropy = self.compute_spatial_entropy(v_proj)  # [B, H, W]
        avg_pool = torch.mean(v_proj, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(v_proj, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_feat = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        spatial_weights = self.spatial_attention(spatial_feat)  # [B, 1, H, W]

        # 3. Mutual Information weights
        mi_weights = self.estimate_mutual_info(v_proj, s_proj)  # [B, C, H, W]

        # Apply all weights
        channel_weighted = v_proj * channel_weights.view(B, C, 1, 1)
        spatial_weighted = channel_weighted * spatial_weights
        final_output = spatial_weighted * mi_weights + visual_feat

        return final_output


def tet_info_fusion():
    # Create random input tensors
    B, C, H, W = 16, 64, 256, 256

    visual_feat = torch.randn(B, C, H, W)
    semantic_feat = torch.randn(B, C, H, W)  # Same shape as visual_feat

    # Initialize module
    fusion_module = InfoFusionModule(in_channels=C)

    # Forward pass
    output = fusion_module(visual_feat, semantic_feat)

    print("Input visual feature shape:", visual_feat.shape)
    print("Input semantic feature shape:", semantic_feat.shape)
    print("Output feature shape:", output.shape)

    # Test entropy computations
    v_channel_entropy = fusion_module.compute_channel_entropy(visual_feat)
    v_spatial_entropy = fusion_module.compute_spatial_entropy(visual_feat)
    print("\nChannel entropy shape:", v_channel_entropy.shape)
    print("Spatial entropy shape:", v_spatial_entropy.shape)

    # Test mutual information estimation
    mi_weights = fusion_module.estimate_mutual_info(visual_feat, semantic_feat)
    print("Mutual information weights shape:", mi_weights.shape)


if __name__ == "__main__":
    tet_info_fusion()