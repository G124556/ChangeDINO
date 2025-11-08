import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 修改投影网络部分的代码
class EntropyMLP(nn.Module):
    def __init__(self,visual_dims=[64, 128, 256, 512], semantic_dim=256):
        super().__init__()
        # 投影网络 - 将视觉特征调整到相同通道数，并添加GN和ReLU
        self.proj_1 = nn.Sequential(
            nn.Conv2d(64, 256, 1),
            nn.GroupNorm(8, 256),  # 使用8个组，通道数为256
            nn.ReLU(inplace=True)
        )
        self.proj_2 = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.proj_3 = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.proj_4 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_level1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_level2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_level3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_level4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )


















        # 熵计算的bin数
        self.num_bins = 20

        # MLP网络 - 为每个尺度设计
        self.mlp_1 = self._build_mlp(512, 64)  # level 1
        self.mlp_2 = self._build_mlp(512, 128)  # level 2
        self.mlp_3 = self._build_mlp(512, 256)  # level 3
        self.mlp_4 = self._build_mlp(512, 512)  # level 4

        self.q_proj = nn.ModuleList([
            nn.Linear(semantic_dim, semantic_dim) for _ in visual_dims
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(semantic_dim, semantic_dim) for _ in visual_dims
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(semantic_dim, semantic_dim) for _ in visual_dims
        ])



    def _build_mlp(self, in_dim, out_channels):
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    # def compute_entropy(self,feature_map):  # feature_map: (H,W)
    #     # 1. 归一化特征图到[0,1]，使其符合概率分布
    #     f = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-6)
    #
    #     # 2. 将特征值转换为概率分布
    #     p = f / (torch.sum(f) + 1e-6)
    #
    #     # 3. 计算信息熵: -Σ(pi * log(pi))
    #     entropy = -torch.sum(p * torch.log2(p + 1e-6))
    #
    #     return entropy

    def compute_entropy(self,feature_map):  # feature_map: (B,C,H,W)
        B, C, H, W = feature_map.shape

        # 创建存储熵值的tensor
        entropy_values = torch.zeros(B, C).to(feature_map.device)

        # 对每个batch和channel计算熵
        for b in range(B):
            for c in range(C):
                # 提取当前channel的特征图 (H,W)
                current_feature = feature_map[b, c]

                # 1. 归一化特征图到[0,1]
                f = (current_feature - current_feature.min()) / (current_feature.max() - current_feature.min() + 1e-6)

                # 2. 将特征值转换为概率分布
                p = f / (torch.sum(f) + 1e-6)

                # 3. 计算信息熵: -Σ(pi * log(pi))
                entropy = -torch.sum(p * torch.log2(p + 1e-6))

                # 存储结果
                entropy_values[b, c] = entropy

        return entropy_values  # 返回形状(B,C)的tensor

    # def cross_attention(self, q, k, v, scale_idx,entrop):
    #     B = q.shape[0]
    #     B, C, H, W = q.shape
    #     # q = q.flatten(2).transpose(1, 2)  # (B, H*W, C)
    #     # k = k.flatten(2).transpose(1, 2)
    #     # v = v.flatten(2).transpose(1, 2)
    #
    #
    #     q = q.flatten(2)  # (B, H*W, C)
    #     k = k.flatten(2)
    #     v = v.flatten(2)
    #
    #
    #     # q = self.q_proj[scale_idx](q)
    #     # k = self.k_proj[scale_idx](k)
    #     # v = self.v_proj[scale_idx](v)
    #
    #     attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(int(H*W))
    #
    #     entrop = (entrop - entrop.min(dim=1, keepdim=True)[0]) / (
    #                 entrop.max(dim=1, keepdim=True)[0] - entrop.min(dim=1, keepdim=True)[0])
    #
    #
    #     attn=attn*(1-entrop.unsqueeze(2))
    #     attn = F.softmax(attn, dim=-1)
    #     return torch.matmul(attn, v)
    def cross_attention(self, q, k, v, scale_idx, entrop):
        B, C, H, W = q.shape

        # 先进行特征展平
        q = q.flatten(2)  # (B, C, H*W)
        k = k.flatten(2)  # (B, C, H*W)
        v = v.flatten(2)  # (B, C, H*W)

        # 添加线性变换层
        # q = self.q_proj[scale_idx](q.transpose(1, 2)).transpose(1, 2)  # (B, C, H*W)
        # k = self.k_proj[scale_idx](k.transpose(1, 2)).transpose(1, 2)
        # v = self.v_proj[scale_idx](v.transpose(1, 2)).transpose(1, 2)

        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(int(H * W))

        # 熵归一化
        entrop = (entrop - entrop.min(dim=1, keepdim=True)[0]) / (
                entrop.max(dim=1, keepdim=True)[0] - entrop.min(dim=1, keepdim=True)[0])

        # 应用熵权重和softmax
        attn = attn * (1 - entrop.unsqueeze(2))
        attn = F.softmax(attn, dim=-1)

        # 计算加权和
        out = torch.matmul(attn, v)

        return out


    def process_level(self, visual_feat, text_feat, mlp, proj,con,i):
        B,C,H,W = visual_feat.shape
        # H, W = visual_feat.shape[2:]

        # 投影视觉特征
        # visual_feat_proj = proj(visual_feat)
        visual_feat_proj = visual_feat

        # 调整文本特征尺寸
        if text_feat.shape[2:] != visual_feat_proj.shape[2:]:
            text_feat = F.interpolate(text_feat, size=visual_feat_proj.shape[2:],
                                      mode='bilinear', align_corners=False)

        # 计算熵
        # entropy_v = self.compute_entropy(visual_feat_proj)  # (B, 256)
        entropy_t = self.compute_entropy(text_feat)  # (B, 256)

        cross=self.cross_attention(visual_feat_proj,text_feat,text_feat,i,entropy_t)

        # 拼接特征
        concat_feat = torch.cat([visual_feat_proj, text_feat], dim=1)  # (B, 512, H, W)

        # 通过MLP生成权重
        # feat_flat = concat_feat.view(B * H * W, 512)
        feat_flat = concat_feat.view(B * H * W, 128)
        MPLweight = mlp(feat_flat).view(B, -1, H, W)

        visual_feat_MLP=con(visual_feat_proj)* MPLweight
        # 应用熵信息
        # entropy_weight =  entropy_t.unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1)
        # weight = weight * entropy_weight

        # 增强原始特征
        enhanced_feat = visual_feat_proj +cross.reshape(B,-1,H,W)+visual_feat_MLP

        return enhanced_feat

    def forward(self, visual_features, text_features):
        # 输入是特征列表，从小尺度到大尺度
        enhanced_features = []

        # Level 1: 64x64
        f1 = self.process_level(visual_features[0], text_features[0],
                                self.mlp_1, self.proj_1,self.conv3x3_level1,0)
        enhanced_features.append(f1)

        # Level 2: 32x32
        f2 = self.process_level(visual_features[1], text_features[1],
                                self.mlp_2, self.proj_2,self.conv3x3_level2,1)
        enhanced_features.append(f2)

        # Level 3: 16x16
        f3 = self.process_level(visual_features[2], text_features[2],
                                self.mlp_3, self.proj_3,self.conv3x3_level3,2)
        enhanced_features.append(f3)

        # Level 4: 8x8
        f4 = self.process_level(visual_features[3], text_features[3],
                                self.mlp_4, self.proj_4,self.conv3x3_level4,3)
        enhanced_features.append(f4)

        return enhanced_features
if __name__ == "__main__":
    # 初始化模型
    model = EntropyMLP()

    # 准备输入特征
    visual_features = [
        torch.randn(16, 64, 64, 64),    # level 1
        torch.randn(16, 128, 32, 32),   # level 2
        torch.randn(16, 256, 16, 16),   # level 3
        torch.randn(16, 512, 8, 8)      # level 4
    ]

    # text_features = [
    #     torch.randn(16, 256, 32, 32),   # level 1
    #     torch.randn(16, 256, 16, 16),   # level 2
    #     torch.randn(16, 256, 8, 8),     # level 3
    #     torch.randn(16, 256, 4, 4)      # level 4
    # ]

    text_features = [
        torch.randn(16, 64, 32, 32),   # level 1
        torch.randn(16, 256, 16, 16),   # level 2
        torch.randn(16, 256, 8, 8),     # level 3
        torch.randn(16, 256, 4, 4)      # level 4
    ]



    # 获取增强后的特征
    enhanced_features = model(visual_features, text_features)
    print(enhanced_features[0].shape)
    print(enhanced_features[1].shape)
    print(enhanced_features[2].shape)
    print(enhanced_features[3].shape)