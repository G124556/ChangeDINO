import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InfoFusionModule(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256, 512], hidden_dim=256):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.hidden_dim = hidden_dim

        # Feature projection networks for each scale
        self.proj_nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, 1),
                nn.GroupNorm(8, hidden_dim),
                nn.ReLU()
            ) for c in in_channels_list
        ])

        # Attention layers for each scale
        self.query_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in in_channels_list
        ])
        self.key_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in in_channels_list
        ])
        self.value_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in in_channels_list
        ])

        # Mutual information estimator
        self.mi_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Conv layers for A_i
        self.conv_A = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )

    def compute_entropy(self, x):
        """Compute information entropy"""
        if len(x.shape) == 4:  # For visual features
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            x = F.softmax(x, dim=2)
            entropy = -torch.sum(x * torch.log(x + 1e-10), dim=2)
            return entropy
        else:  # For semantic features
            x = F.softmax(x, dim=-1)
            entropy = -torch.sum(x * torch.log(x + 1e-10), dim=-1)
            return entropy

    def compute_mutual_info(self, F_i, G_s):
        """Compute mutual information following paper's formula"""
        # Flatten spatial dimensions
        b, c, h, w = F_i.shape
        F_i_flat = F_i.view(b, c, -1).mean(dim=2)  # [B, C]
        G_s_flat = G_s.mean(dim=1)  # [B, C]

        # Concatenate features
        joint = torch.cat([F_i_flat, G_s_flat], dim=1)
        mi = self.mi_estimator(joint)
        return mi

    def compute_attention(self, F_i, G_s, G_s_v, scale_idx):
        """Compute attention following paper's Attn(F_i, G_s, G_s) formula"""
        b, c, h, w = F_i.shape
        F_i_flat = F_i.view(b, c, -1).permute(0, 2, 1)  # [B, HW, C]

        Q = self.query_projs[scale_idx](F_i_flat)
        K = self.key_projs[scale_idx](G_s)
        V = self.value_projs[scale_idx](G_s_v)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(c)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)

        return out.permute(0, 2, 1).view(b, c, h, w)

    def forward(self, visual_feats, semantic_feats):
        """
        Args:
            visual_feats: List of tensors [B, Ci, Hi, Wi]
            semantic_feats: List of tensors [B, N, D]
        """
        outputs = []

        for i, (F_i, G_s) in enumerate(zip(visual_feats, semantic_feats)):
            # Feature projection
            P_i = self.proj_nets[i](F_i)

            # 1. Compute entropy weights
            w_i = F.softmax(self.compute_entropy(P_i) + self.compute_entropy(G_s))

            # 2. Compute attention
            A_i = self.conv_A(
                torch.cat([P_i, G_s.view(*G_s.shape[:2], 1, 1).expand(-1, -1, P_i.shape[2], P_i.shape[3])], dim=1))
            attn_out = self.compute_attention(F_i, G_s, G_s, i)

            # 3. Compute F_e,i
            F_e_i = F_i + w_i.view(-1, 1, 1, 1) * (attn_out * A_i)

            # 4. Compute mutual information and final fusion
            alpha_i = self.compute_mutual_info(F_i, G_s)
            mi = alpha_i.view(-1, 1, 1, 1)

            # 5. Final fusion following paper's formula
            F_f_i = mi * F_e_i + (1 - mi) * F_i

            outputs.append(F_f_i)

        return outputs


def tet_info_fusion():
    # 生成多尺度特征
    B = 16
    N = 100  # 语义特征的token数量
    D = 256  # 隐藏维度

    scales = [
        (64, 256, 256),  # C1, H1, W1
        (128, 128, 128),  # C2, H2, W2
        (256, 64, 64),  # C3, H3, W3
        (512, 32, 32)  # C4, H4, W4
    ]

    visual_feats = [torch.randn(B, c, h, w) for c, h, w in scales]
    semantic_feats = [torch.randn(B, N, D) for _ in scales]

    # 初始化模块
    fusion_module = InfoFusionModule([c for c, _, _ in scales], hidden_dim=D)

    # 前向传播
    outputs = fusion_module(visual_feats, semantic_feats)

    # 打印每个尺度的特征形状
    print("\nFeature shapes at each scale:")
    for i, output in enumerate(outputs):
        print(f"Scale {i + 1}: {output.shape}")

    print("\nTesting mutual information computation:")
    mi = fusion_module.compute_mutual_info(visual_feats[0], semantic_feats[0])
    print(f"Mutual information shape: {mi.shape}")

    print("\nTesting entropy computation:")
    entropy_v = fusion_module.compute_entropy(visual_feats[0])
    entropy_s = fusion_module.compute_entropy(semantic_feats[0])
    print(f"Visual entropy shape: {entropy_v.shape}")
    print(f"Semantic entropy shape: {entropy_s.shape}")


if __name__ == "__main__":
    tet_info_fusion()