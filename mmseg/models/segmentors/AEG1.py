import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyGuidedAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, visual_feat, text_feat):
        B, D, H, W = visual_feat.shape
        _, Q, _ = text_feat.shape

        visual_feat_flat = visual_feat.flatten(2).permute(0, 2, 1)  # [B, HW, D]

        visual_entropy = self.compute_entropy(visual_feat_flat)  # [B, HW]
        text_entropy = self.compute_entropy(text_feat)  # [B, Q]

        q = self.query(visual_feat_flat)  # [B, HW, D]
        k = self.key(text_feat)  # [B, Q, D]
        v = self.value(text_feat)  # [B, Q, D]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)  # [B, HW, Q]

        entropy_weights = torch.matmul(visual_entropy.unsqueeze(-1),
                                       text_entropy.unsqueeze(1))  # [B, HW, Q]
        attn = attn * entropy_weights
        attn = F.softmax(attn, dim=-1)

        enhanced = torch.matmul(attn, v)  # [B, HW, D]
        enhanced = enhanced.permute(0, 2, 1).reshape(B, D, H, W)

        return enhanced

    def compute_entropy(self, x):
        x = F.softmax(x, dim=-1)
        entropy = -torch.sum(x * torch.log(x + 1e-9), dim=-1)
        return F.softmax(entropy, dim=-1)


class AfterExtractFeatDino(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.text_projection = nn.ModuleList([
            nn.Linear(256, hidden_dim) for _ in range(4)
        ])

        self.attention_modules = nn.ModuleList([
            EntropyGuidedAttention(hidden_dim) for _ in range(4)
        ])

        # 修改final_projection以产生正确的输出维度
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)  # 128 = 2 * 8 * 8
        )

    def forward(self, xA_list, memory_textA):
        batch_size = xA_list[0].shape[0]
        enhanced_features = []

        for i, x in enumerate(xA_list[:4]):
            text_feat = self.text_projection[i](memory_textA)
            enhanced_feat = self.attention_modules[i](x, text_feat)
            final_feat = enhanced_feat + x
            enhanced_features.append(final_feat)

        # 修改最终特征处理
        final_feature = enhanced_features[-1].mean(dim=[2, 3])  # [B, D]
        text_embeddingsA = final_feature.unsqueeze(1).repeat(1, 2, 1)  # [B, 2, D]

        # 修改score_mapA的生成方式
        score_map_flat = self.final_projection(final_feature)  # [B, 128]
        score_mapA = score_map_flat.view(batch_size, 2, 8, 8)  # [B, 2, 8, 8]

        return text_embeddingsA, enhanced_features, score_mapA


def tet_after_extract_feat_dino():
    torch.manual_seed(42)

    batch_size = 4
    hidden_dim = 256

    xA_list = [
        torch.randn(batch_size, hidden_dim, 64, 64),
        torch.randn(batch_size, hidden_dim, 32, 32),
        torch.randn(batch_size, hidden_dim, 16, 16),
        torch.randn(batch_size, hidden_dim, 8, 8),
    ]
    memory_textA = torch.randn(batch_size, 4, 256)

    model = AfterExtractFeatDino(hidden_dim=hidden_dim)
    model.eval()

    with torch.no_grad():
        text_embeddingsA, enhanced_features, score_mapA = model(xA_list, memory_textA)

    print("\n=== 输出形状检查 ===")
    print(f"文本嵌入形状: {text_embeddingsA.shape}")
    print("\n增强特征形状:")
    for i, feat in enumerate(enhanced_features):
        print(f"Level {i}: {feat.shape}")
    print(f"\n分数图形状: {score_mapA.shape}")

    print("\n=== 数值范围检查 ===")
    print(f"文本嵌入范围: [{text_embeddingsA.min():.3f}, {text_embeddingsA.max():.3f}]")
    print(f"分数图范围: [{score_mapA.min():.3f}, {score_mapA.max():.3f}]")

    for i, feat in enumerate(enhanced_features):
        print(f"Level {i} - 范围: [{feat.min():.3f}, {feat.max():.3f}], 均值: {feat.mean():.3f}")

    assert text_embeddingsA.shape == (batch_size, 2, hidden_dim)
    assert len(enhanced_features) == 4
    assert score_mapA.shape == (batch_size, 2, 8, 8)

    print("\n=== 测试通过! ===")

    return text_embeddingsA, enhanced_features, score_mapA


if __name__ == "__main__":
    text_embeddingsA, enhanced_features, score_mapA = tet_after_extract_feat_dino()