import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InfoFusionModule(nn.Module):
    def __init__(self, visual_dims=[64, 128, 256, 512], semantic_dim=256):
        super().__init__()
        self.visual_dims = visual_dims
        self.semantic_dim = semantic_dim

        # Visual feature projection for each scale
        self.visual_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, semantic_dim, 1),
                nn.GroupNorm(8, semantic_dim),
                nn.ReLU()
            ) for dim in visual_dims
        ])

        # MLP for entropy computation
        self.entropy_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(semantic_dim, semantic_dim // 2),
                nn.ReLU(),
                nn.Linear(semantic_dim // 2, semantic_dim // 4),
                nn.ReLU(),
                nn.Linear(semantic_dim // 4, 1)
            ) for _ in visual_dims
        ])

        # Mutual information estimator for each scale
        self.mi_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(semantic_dim * 2, semantic_dim),
                nn.ReLU(),
                nn.Linear(semantic_dim, 1),
                nn.Sigmoid()
            ) for _ in visual_dims
        ])

        # Cross attention layers
        self.q_proj = nn.ModuleList([
            nn.Linear(semantic_dim, semantic_dim) for _ in visual_dims
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(semantic_dim, semantic_dim) for _ in visual_dims
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(semantic_dim, semantic_dim) for _ in visual_dims
        ])

        # Output projection to restore original channels
        self.out_projs = nn.ModuleList([
            nn.Conv2d(semantic_dim, dim, 1) for dim in visual_dims
        ])

    def compute_entropy(self, x, scale_idx):
        B = x.shape[0]
        # Convert to probability distribution
        if len(x.shape) == 4:  # For spatial features
            x = x.view(B, self.semantic_dim, -1)
            x = F.softmax(x, dim=-1)
            x = x.mean(dim=-1)  # [B, C]
        else:  # For semantic features
            x = F.softmax(x, dim=-1)
            x = x.mean(dim=-1)  # [B, C]

        return self.entropy_mlp[scale_idx](x)  # [B, 1]

    def compute_mutual_info(self, v_feat, s_feat, scale_idx):
        B = v_feat.shape[0]
        # Mean pooling over spatial dimensions
        v_feat = v_feat.view(B, self.semantic_dim, -1).mean(-1)  # [B, C]
        s_feat = s_feat.view(B, self.semantic_dim, -1).mean(-1)  # [B, C]

        joint = torch.cat([v_feat, s_feat], dim=1)  # [B, 2C]
        return self.mi_estimator[scale_idx](joint)  # [B, 1]

    def cross_attention(self, q, k, v, scale_idx):
        B = q.shape[0]
        q = self.q_proj[scale_idx](q)
        k = self.k_proj[scale_idx](k)
        v = self.v_proj[scale_idx](v)

        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.semantic_dim)
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, visual_feats, semantic_feats):
        outputs = []

        for i, (v_feat, s_feat) in enumerate(zip(visual_feats, semantic_feats)):
            # Project visual features to semantic dimension
            v_proj = self.visual_projs[i](v_feat)  # [B, semantic_dim, H, W]

            # Compute entropies
            v_entropy = self.compute_entropy(v_proj, i)
            s_entropy = self.compute_entropy(s_feat, i)
            entropy_weight = F.softmax(v_entropy + s_entropy, dim=1)

            # Prepare for attention
            B, C, H, W = v_proj.shape
            v_flat = v_proj.flatten(2).transpose(1, 2)  # [B, HW, C]
            s_flat = s_feat.flatten(2).transpose(1, 2)  # [B, HW, C]

            # Cross attention
            attn_out = self.cross_attention(v_flat, s_flat, s_flat, i)
            attn_out = attn_out.transpose(1, 2).view(B, C, H, W)

            # Compute mutual information
            mi_weight = self.compute_mutual_info(v_proj, s_feat, i)
            mi_weight = mi_weight.view(B, 1, 1, 1)

            # Enhanced features
            enhanced = v_proj + entropy_weight.view(B, 1, 1, 1) * attn_out

            # Final fusion
            fused = mi_weight * enhanced + (1 - mi_weight) * v_proj

            # Project back to original channel dimension
            out = self.out_projs[i](fused)

            outputs.append(out)

        return outputs


def tet_info_fusion():
    # Visual feature dimensions: [B, C, H, W]
    v_feat1 = torch.randn(16, 64, 64, 64)
    v_feat2 = torch.randn(16, 128, 32, 32)
    v_feat3 = torch.randn(16, 256, 16, 16)
    v_feat4 = torch.randn(16, 512, 8, 8)
    visual_feats = [v_feat1, v_feat2, v_feat3, v_feat4]

    # Semantic feature dimensions: [B, C, H, W]
    s_feat1 = torch.randn(16, 256, 32, 32)
    s_feat2 = torch.randn(16, 256, 16, 16)
    s_feat3 = torch.randn(16, 256, 8, 8)
    s_feat4 = torch.randn(16, 256, 4, 4)
    semantic_feats = [s_feat1, s_feat2, s_feat3, s_feat4]

    # Initialize module
    fusion_module = InfoFusionModule()

    # Forward pass
    outputs = fusion_module(visual_feats, semantic_feats)

    # Print shapes
    print("\nInput feature shapes:")
    for i, (v, s) in enumerate(zip(visual_feats, semantic_feats)):
        print(f"Scale {i + 1}:")
        print(f"  Visual: {v.shape}")
        print(f"  Semantic: {s.shape}")

    print("\nOutput feature shapes:")
    for i, out in enumerate(outputs):
        print(f"Scale {i + 1}: {out.shape}")


if __name__ == "__main__":
    tet_info_fusion()