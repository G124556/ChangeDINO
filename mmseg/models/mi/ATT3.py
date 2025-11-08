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

        # 语义生成器
        self.semantic_generator = nn.Sequential(
            nn.Conv2d(text_dim * 4, text_dim, 1),
            nn.BatchNorm2d(text_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(text_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * 1024)
        )

    def forward(self, vision_features, text_features):
        if isinstance(vision_features, tuple):
            vision_features = list(vision_features)
        if isinstance(text_features, tuple):
            text_features = list(text_features)

        batch_size = vision_features[0].shape[0]
        enhanced_features = []

        # 并行处理所有尺度的特征
        for i, (v_feat, t_feat, v_proj, out_proj) in enumerate(zip(
                vision_features, text_features, self.vision_projections,
                self.out_projections)):
            # 1. 投影视觉特征到文本特征空间
            v_proj_feat = v_proj(v_feat)

            # 2. 计算QKV
            q = self.q_proj(v_proj_feat)
            k = self.k_proj(t_feat)
            v = self.v_proj(t_feat)

            # 3. 计算注意力和特征增强
            B, C, H, W = v_proj_feat.shape
            q_flat = q.flatten(2)  # B, C, HW
            k_flat = k.flatten(2)  # B, C, H'W'
            v_flat = v.flatten(2)  # B, C, H'W'

            attn = torch.matmul(q_flat.transpose(1, 2), k_flat) / (self.text_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)  # B, HW, H'W'

            # 4. 使用注意力增强特征
            attended_value = torch.matmul(v_flat, attn.transpose(1, 2))  # B, C, HW
            attended_feat = attended_value.view(B, C, H, W)

            # 5. 投影回原始维度并做残差连接
            enhanced = out_proj(attended_feat) + v_feat
            enhanced_features.append(enhanced)

        # 生成语义信息
        aligned_text_features = []
        smallest_size = min(feat.shape[2:] for feat in text_features)

        for feat in text_features:
            if feat.shape[2:] != smallest_size:
                feat = F.adaptive_avg_pool2d(feat, smallest_size)
            aligned_text_features.append(feat)

        concat_features = torch.cat(aligned_text_features, dim=1)
        semantic_info = self.semantic_generator(concat_features)
        semantic_info = semantic_info.view(batch_size, 2, 1024)

        return  semantic_info,enhanced_features


# 测试代码
def tet_cross_attention():
    # 创建模型
    model = CrossAttentionModule()
    model = model.cuda()  # 移动到GPU

    # 创建测试数据
    vision_features = [
        torch.randn(16, 64, 64, 64).cuda(),  # 第一层
        torch.randn(16, 128, 32, 32).cuda(),  # 第二层
        torch.randn(16, 256, 16, 16).cuda(),  # 第三层
        torch.randn(16, 512, 8, 8).cuda()  # 第四层
    ]

    text_features = [
        torch.randn(16, 256, 32, 32).cuda(),  # 第一层
        torch.randn(16, 256, 16, 16).cuda(),  # 第二层
        torch.randn(16, 256, 8, 8).cuda(),  # 第三层
        torch.randn(16, 256, 4, 4).cuda()  # 第四层
    ]

    # 前向传播
    enhanced_features, semantic_info = model(vision_features, text_features)

    # 打印输出形状
    print("Enhanced features shapes:")
    for i, feat in enumerate(enhanced_features):
        print(f"Layer {i + 1}:", feat.shape)
    print("Semantic info shape:", semantic_info.shape)

    return enhanced_features, semantic_info


if __name__ == "__main__":
    enhanced_features, semantic_info = tet_cross_attention()