import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleEntropyGuidance(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 为每个尺度创建独立的query投影
        self.query_projs = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(4)
        ])
        # 共享的key和value投影
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def compute_entropy(self, x):
        """计算特征熵"""
        x = F.softmax(x, dim=-1)
        entropy = -torch.sum(x * torch.log(x + 1e-9), dim=-1)
        return F.softmax(entropy, dim=-1)

    def forward(self, visual_feats, text_feat):
        """
        Args:
            visual_feats: List[Tensor], 每个tensor形状为[B, D, Hi, Wi]
            text_feat: [B, Q, H] 文本特征
        Returns:
            enhanced_feats: List[Tensor], 增强后的多尺度特征
        """
        batch_size = visual_feats[0].shape[0]
        enhanced_feats = []

        # 处理文本特征
        k = self.key(text_feat)  # [B, Q, D]
        v = self.value(text_feat)  # [B, Q, D]
        text_entropy = self.compute_entropy(text_feat)  # [B, Q]

        # 逐尺度处理
        for i, feat in enumerate(visual_feats):
            B, D, H, W = feat.shape

            # 1. 展平视觉特征
            vis_flat = feat.flatten(2).permute(0, 2, 1)  # [B, HW, D]

            # 2. 计算视觉特征熵
            vis_entropy = self.compute_entropy(vis_flat)  # [B, HW]

            # 3. 生成query
            q = self.query_projs[i](vis_flat)  # [B, HW, D]

            # 4. 计算注意力分数
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)  # [B, HW, Q]

            # 5. 使用熵权重调制注意力
            entropy_map = torch.matmul(vis_entropy.unsqueeze(-1),
                                       text_entropy.unsqueeze(1))  # [B, HW, Q]
            attn = attn * entropy_map
            attn = F.softmax(attn, dim=-1)

            # 6. 计算增强特征
            enhanced = torch.matmul(attn, v)  # [B, HW, D]
            enhanced = enhanced.permute(0, 2, 1).reshape(B, D, H, W)

            # 7. 残差连接
            enhanced = enhanced + feat
            enhanced_feats.append(enhanced)

        return enhanced_feats


# 测试代码
def tet_multi_scale_guidance():
    batch_size = 4
    dim = 256
    Q = 8  # text序列长度

    # 创建测试数据
    visual_feats = [
        torch.randn(batch_size, dim, 64, 64),
        torch.randn(batch_size, dim, 32, 32),
        torch.randn(batch_size, dim, 16, 16),
        torch.randn(batch_size, dim, 8, 8)
    ]
    text_feat = torch.randn(batch_size, Q, dim)

    # 创建模型
    model = MultiScaleEntropyGuidance(dim)

    # 前向传播
    enhanced_feats = model(visual_feats, text_feat)

    # 打印结果
    print("\n=== 输出检查 ===")
    for i, feat in enumerate(enhanced_feats):
        print(f"Level {i} shape: {feat.shape}")
        print(f"Level {i} range: [{feat.min():.3f}, {feat.max():.3f}]")
        print(f"Level {i} mean: {feat.mean():.3f}")
        print()

        # 验证维度没有改变
        assert feat.shape == visual_feats[i].shape

    print("测试通过!")


if __name__ == "__main__":
    tet_multi_scale_guidance()