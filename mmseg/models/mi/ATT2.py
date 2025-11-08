import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelCrossAttention(nn.Module):
    def __init__(self, text_dim=256):
        super().__init__()
        self.text_dim = text_dim

        # 视觉特征投影层（不同尺度）
        self.vision_projections = nn.ModuleList([
            nn.Conv2d(64, text_dim, 1),  # 第一层
            nn.Conv2d(128, text_dim, 1),  # 第二层
            nn.Conv2d(256, text_dim, 1),  # 第三层
            nn.Conv2d(512, text_dim, 1)  # 第四层
        ])

        # QKV投影层（共享权重）
        self.q_proj = nn.Conv2d(text_dim, text_dim, 1)
        self.k_proj = nn.Conv2d(text_dim, text_dim, 1)
        self.v_proj = nn.Conv2d(text_dim, text_dim, 1)

        # 输出投影层（不同尺度）
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

        # 1. 并行处理所有视觉特征
        aligned_vision = []
        for feat, proj in zip(vision_features, self.vision_projections):
            aligned_vision.append(proj(feat))

        # 2. 并行计算所有QKV
        queries = []
        keys = []
        values = []
        for v_feat, t_feat in zip(aligned_vision, text_features):
            # 保存原始形状
            B, C, H, W = v_feat.shape
            _, _, Ht, Wt = t_feat.shape

            # 计算QKV
            q = self.q_proj(v_feat).flatten(2)  # B, C, HW
            k = self.k_proj(t_feat).flatten(2)  # B, C, HtWt
            v = self.v_proj(t_feat).flatten(2)  # B, C, HtWt

            queries.append((q, H, W))
            keys.append(k)
            values.append(v)

        # 3. 并行计算注意力
        enhanced_features = []
        for idx, ((q, H, W), k, v, out_proj) in enumerate(zip(queries, keys, values, self.out_projections)):
            # 计算注意力分数
            attn = torch.matmul(q.transpose(1, 2), k) / (self.text_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)

            # 应用注意力
            out = torch.matmul(attn, v.transpose(1, 2))  # B, HW, C
            out = out.transpose(1, 2).view(batch_size, self.text_dim, H, W)

            # 投影回原始维度并添加残差连接
            enhanced = out_proj(out) + vision_features[idx]
            enhanced_features.append(enhanced)

        # 4. 生成语义信息
        # 首先将所有文本特征调整到相同大小
        aligned_text_features = []
        smallest_size = min(feat.shape[2:] for feat in text_features)
        for feat in text_features:
            if feat.shape[2:] != smallest_size:
                feat = F.adaptive_avg_pool2d(feat, smallest_size)
            aligned_text_features.append(feat)

        # 拼接所有文本特征
        concat_features = torch.cat(aligned_text_features, dim=1)
        semantic_info = self.semantic_generator(concat_features)
        semantic_info = semantic_info.view(batch_size, 2, 1024)

        # return enhanced_features, semantic_info
        return semantic_info,enhanced_features


if __name__ == "__main__":
    # 创建模型
    model = ParallelCrossAttention()

    # 准备输入
    vision_features = [
        torch.randn(16, 64, 64, 64),    # 第一层特征
        torch.randn(16, 128, 32, 32),   # 第二层特征
        torch.randn(16, 256, 16, 16),   # 第三层特征
        torch.randn(16, 512, 8, 8)      # 第四层特征
    ]

    text_features = [
        torch.randn(16, 256, 32, 32),   # 第一层文本特征
        torch.randn(16, 256, 16, 16),   # 第二层文本特征
        torch.randn(16, 256, 8, 8),     # 第三层文本特征
        torch.randn(16, 256, 4, 4)      # 第四层文本特征
    ]

    # 一次性处理所有特征
    guided_feats, semantic_info = model(vision_features, text_features)

    B = 16

    assert semantic_info.shape == (B, 2, 1024), f"语义信息维度错误: {semantic_info.shape}"
    assert len(guided_feats) == 4, "特征数量错误"
    assert guided_feats[0].shape == (B, 64, 64, 64), f"特征0维度错误: {guided_feats[0].shape}"
    assert guided_feats[1].shape == (B, 128, 32, 32), f"特征1维度错误: {guided_feats[1].shape}"
    assert guided_feats[2].shape == (B, 256, 16, 16), f"特征2维度错误: {guided_feats[2].shape}"
    assert guided_feats[3].shape == (B, 512, 8, 8), f"特征3维度错误: {guided_feats[3].shape}"

    print("\n测试通过!")
