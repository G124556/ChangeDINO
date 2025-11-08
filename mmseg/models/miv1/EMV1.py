import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InfoFusionModule(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Feature projection networks
        self.visual_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU()
        )

        self.semantic_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mutual information estimator (implemented as a MLP)
        self.mi_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Cross attention layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

    def compute_entropy(self, x):
        # Convert features to probability distributions using softmax
        if len(x.shape) == 4:  # For visual features
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            x = F.softmax(x, dim=2)
            entropy = -torch.sum(x * torch.log(x + 1e-10), dim=2)
            return entropy.mean(dim=1)
        else:  # For semantic features
            x = F.softmax(x, dim=-1)
            entropy = -torch.sum(x * torch.log(x + 1e-10), dim=-1)
            return entropy.mean(dim=-1)

    def estimate_mutual_info(self, f_v, f_s):
        # Create positive and negative pairs
        batch_size = f_v.shape[0]

        # Reshape visual features
        f_v = f_v.view(batch_size, self.hidden_dim, -1).mean(dim=-1)  # [B, D]

        # Mean pool semantic features
        f_s = f_s.mean(dim=1)  # [B, D]

        # Concatenate features
        joint = torch.cat([f_v, f_s], dim=1)  # [B, 2D]

        # Create negative pairs by shuffling the batch
        f_s_shuffle = f_s[torch.randperm(batch_size)]
        marginal = torch.cat([f_v, f_s_shuffle], dim=1)  # [B, 2D]

        # Estimate mutual information
        joint_score = self.mi_estimator(joint)
        marginal_score = self.mi_estimator(marginal)

        mi_lb = torch.mean(joint_score) - torch.log(torch.mean(torch.exp(marginal_score)))
        return torch.sigmoid(joint_score)  # Return probabilities for weighting

    def forward(self, visual_feat, semantic_feat):
        """
        Args:
            visual_feat: [B, C, H, W]
            semantic_feat: [B, N, D]
        """
        # Project features
        v_proj = self.visual_proj(visual_feat)  # [B, D, H, W]
        s_proj = self.semantic_proj(semantic_feat)  # [B, N, D]

        # Compute entropies
        v_entropy = self.compute_entropy(v_proj)  # [B]
        s_entropy = self.compute_entropy(s_proj)  # [B]

        # Compute mutual information weights
        mi_weights = self.estimate_mutual_info(v_proj, s_proj)  # [B, 1]

        # Compute attention weights based on entropy
        entropy_weights = F.softmax(torch.stack([v_entropy, s_entropy], dim=1), dim=1)

        # Cross attention
        B, D, H, W = v_proj.shape
        v_flat = v_proj.flatten(2).transpose(1, 2)  # [B, HW, D]

        q = self.q_proj(v_flat)  # [B, HW, D]
        k = self.k_proj(s_proj)  # [B, N, D]
        v = self.v_proj(s_proj)  # [B, N, D]

        attn = (q @ k.transpose(-2, -1)) / np.sqrt(D)  # [B, HW, N]
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v)  # [B, HW, D]
        out = out.transpose(1, 2).view(B, D, H, W)  # [B, D, H, W]

        # Final fusion with dynamic weights
        final_output = mi_weights.view(B, 1, 1, 1) * out + \
                       (1 - mi_weights.view(B, 1, 1, 1)) * visual_feat

        return final_output


# Test the module
def tet_info_fusion():
    # Create random input tensors
    B, C, H, W = 16, 64, 256, 256
    N, D = 100, 128  # N: number of semantic tokens, D: hidden dimension

    visual_feat = torch.randn(B, C, H, W)
    semantic_feat = torch.randn(B, N, D)

    # Initialize module
    fusion_module = InfoFusionModule(in_channels=C, hidden_dim=D)

    # Forward pass
    output = fusion_module(visual_feat, semantic_feat)

    print("Input visual feature shape:", visual_feat.shape)
    print("Input semantic feature shape:", semantic_feat.shape)
    print("Output feature shape:", output.shape)

    # Test entropy computation
    v_entropy = fusion_module.compute_entropy(visual_feat)
    s_entropy = fusion_module.compute_entropy(semantic_feat)
    print("\nVisual entropy shape:", v_entropy.shape)
    print("Semantic entropy shape:", s_entropy.shape)

    # Test mutual information estimation
    mi_weights = fusion_module.estimate_mutual_info(
        fusion_module.visual_proj(visual_feat),
        fusion_module.semantic_proj(semantic_feat)
    )
    print("Mutual information weights shape:", mi_weights.shape)


if __name__ == "__main__":
    tet_info_fusion()