import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.projection(x)


class SpatialAttention(nn.Module):
    def __init__(self, pos_dim, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, pos_feat):
        # pos_feat: (B, N, D) -> attention: (B, N, 1)
        return self.attention(pos_feat)


class FiLMModulation(nn.Module):
    def __init__(self, visual_channels, semantic_dim=1024, pos_dim=256):
        super().__init__()
        self.visual_channels = visual_channels  # [64, 128, 256, 512]

        # 语义特征投影
        self.semantic_projections = nn.ModuleList([
            FeatureProjection(semantic_dim, 256) for _ in range(4)
        ])

        # 位置特征注意力
        self.pos_attention = SpatialAttention(pos_dim, 128)

        # FiLM参数生成器
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, visual_ch * 2),  # 生成gamma和beta
                nn.ReLU()
            ) for visual_ch in visual_channels
        ])

        # 输出投影
        self.output_projections = nn.ModuleList([
            nn.Conv2d(ch, 256, 1) for ch in visual_channels
        ])

    def forward(self, visual_feats, semantic_feat, pos_feat):
        """
        Args:
            visual_feats: List of tensors [(B,C1,H1,W1), (B,C2,H2,W2), ...]
            semantic_feat: (B, 2, 1024)
            pos_feat: (B, 900, 256)
        """
        B = visual_feats[0].size(0)

        # 处理语义特征
        semantic_feat = semantic_feat.mean(dim=1)  # (B, 1024)
        semantic_projs = [proj(semantic_feat) for proj in self.semantic_projections]  # [(B, 256), ...]

        # 处理位置特征
        pos_weights = self.pos_attention(pos_feat)  # (B, 900, 1)
        pos_feat = torch.sum(pos_feat * pos_weights, dim=1)  # (B, 256)

        modulated_feats = []
        for i, (visual_feat, semantic_proj) in enumerate(zip(visual_feats, semantic_projs)):
            # 结合位置和语义特征
            combined_feat = semantic_proj + pos_feat  # (B, 256)

            # 生成FiLM参数
            film_params = self.film_generators[i](combined_feat)  # (B, C*2)
            C = visual_feat.size(1)
            gamma, beta = film_params.chunk(2, dim=1)  # 各(B, C)

            # 重塑参数
            gamma = gamma.view(B, C, 1, 1)
            beta = beta.view(B, C, 1, 1)

            # 应用FiLM调制
            modulated = gamma * visual_feat + beta

            # 投影到统一通道数
            output = self.output_projections[i](modulated)
            modulated_feats.append(output)

        return modulated_feats


class ChangeDetectionModule(nn.Module):
    def __init__(self):
        super().__init__()
        visual_channels = [64, 128, 256, 512]
        self.film_modulation = FiLMModulation(visual_channels)

    def forward(self, VA, VB, GA, GB, PA, PB):
        """
        Args:
            VA, VB: List of visual features [(B,C1,H1,W1), ...]
            GA, GB: Semantic features (B,2,1024)
            PA, PB: Position features (B,900,256)
        """
        # 对变化前后的特征分别进行调制
        modulated_A = self.film_modulation(VA, GA, PA)
        modulated_B = self.film_modulation(VB, GB, PB)

        # 计算差异
        changes = []
        for feat_A, feat_B in zip(modulated_A, modulated_B):
            change = torch.abs(feat_A - feat_B)
            changes.append(change)

        return changes


# 使用示例
def main():
    # 构造示例输入
    batch_size = 16
    VA = [
        torch.randn(batch_size, 64, 64, 64),
        torch.randn(batch_size, 128, 32, 32),
        torch.randn(batch_size, 256, 16, 16),
        torch.randn(batch_size, 512, 8, 8)
    ]
    VB = [torch.randn_like(v) for v in VA]
    GA = torch.randn(batch_size, 2, 1024)
    GB = torch.randn_like(GA)
    PA = torch.randn(batch_size, 900, 256)
    PB = torch.randn_like(PA)

    # 初始化模型
    model = ChangeDetectionModule()

    # 前向传播
    changes = model(VA, VB, GA, GB, PA, PB)

    # 验证输出尺寸
    expected_shapes = [
        (batch_size, 256, 64, 64),
        (batch_size, 256, 32, 32),
        (batch_size, 256, 16, 16),
        (batch_size, 256, 8, 8)
    ]

    for change, expected_shape in zip(changes, expected_shapes):
        assert change.shape == expected_shape

    print("All output shapes match expected dimensions!")

main()