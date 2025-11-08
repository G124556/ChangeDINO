import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionModule(nn.Module):
    def __init__(self, text_dim=256):
        super().__init__()
        self.text_dim = text_dim

        # 视觉特征投影层
        self.vision_projections = nn.ModuleList([
            nn.Conv2d(64, text_dim, 1),  # 第一层
            nn.Conv2d(128, text_dim, 1),  # 第二层
            nn.Conv2d(256, text_dim, 1),  # 第三层
            nn.Conv2d(512, text_dim, 1)  # 第四层
        ])

        # QKV投影层
        self.q_proj = nn.Conv2d(text_dim, text_dim, 1)
        self.k_proj = nn.Conv2d(text_dim, text_dim, 1)
        self.v_proj = nn.Conv2d(text_dim, text_dim, 1)

        # 输出投影层
        self.out_projections = nn.ModuleList([
            nn.Conv2d(text_dim, 64, 1),  # 第一层
            nn.Conv2d(text_dim, 128, 1),  # 第二层
            nn.Conv2d(text_dim, 256, 1),  # 第三层
            nn.Conv2d(text_dim, 512, 1)  # 第四层
        ])

    def forward(self, vision_features, text_features):
        if isinstance(vision_features, tuple):
            vision_features = list(vision_features)
        if isinstance(text_features, tuple):
            text_features = list(text_features)

        enhanced_features = []

        # 处理每个尺度的特征
        for v_feat, t_feat, v_proj, out_proj in zip(
                vision_features, text_features, self.vision_projections, self.out_projections):
            # 1. 投影视觉特征
            v_feat = v_proj(v_feat)

            # 2. 生成Q、K、V
            Q = self.q_proj(v_feat)
            K = self.k_proj(t_feat)
            V = self.v_proj(t_feat)

            # 3. 计算注意力图并进行缩放
            attention_map = torch.matmul(Q.flatten(2).transpose(-2, -1),
                                         K.flatten(2)) / (self.text_dim ** 0.5)
            attention_map = F.softmax(attention_map, dim=-1)

            # 4. 应用注意力获取增强特征
            enhanced = torch.matmul(attention_map,
                                    V.flatten(2)).transpose(-2, -1).reshape_as(V)

            # 5. 投影回原始维度并进行残差连接
            enhanced = out_proj(enhanced) + v_feat
            enhanced_features.append(enhanced)

        return enhanced_features


def tet_attention_module():
    # 创建模型
    model = CrossAttentionModule()
    model = model.cuda()

    # 创建测试数据
    vision_features = [
        torch.randn(16, 64, 64, 64).cuda(),
        torch.randn(16, 128, 32, 32).cuda(),
        torch.randn(16, 256, 16, 16).cuda(),
        torch.randn(16, 512, 8, 8).cuda()
    ]

    text_features = [
        torch.randn(16, 256, 32, 32).cuda(),
        torch.randn(16, 256, 16, 16).cuda(),
        torch.randn(16, 256, 8, 8).cuda(),
        torch.randn(16, 256, 4, 4).cuda()
    ]

    # 前向传播
    enhanced_features = model(vision_features, text_features)

    # 打印输出形状
    print("\nEnhanced feature shapes:")
    for i, feat in enumerate(enhanced_features):
        print(f"Layer {i + 1}: {feat.shape}")

    return enhanced_features


if __name__ == "__main__":
    enhanced_features = tet_attention_module()