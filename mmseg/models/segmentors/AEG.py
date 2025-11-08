import torch
import torch.nn as nn
import torch.nn.functional as F
class MutualInformationEstimator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, y):
        batch_size = x.size(0)
        shuffle_index = torch.randperm(batch_size)
        joint = self.net(torch.cat([x, y], dim=-1))
        marginal = self.net(torch.cat([x, y[shuffle_index]], dim=-1))
        return (joint.mean() - torch.log(marginal.exp().mean() + 1e-8)).squeeze()


class EntropyGuidedAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, visual, textual, entropy_weights):
        q = self.query(visual)
        k = self.key(textual)
        v = self.value(textual)

        attn = (q @ k.transpose(-2, -1)) / (visual.size(-1) ** 0.5)
        attn = attn * entropy_weights.unsqueeze(1)
        attn = F.softmax(attn, dim=-1)

        return attn @ v
class AfterExtractFeatDino(nn.Module):
    def __init__(self, hidden_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 视觉特征投影
        self.visual_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, hidden_dim, 1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, hidden_dim, 1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(256, hidden_dim, 1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, hidden_dim, 1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU()
            )
        ])

        # 文本特征处理
        self.text_projection = nn.Linear(256, hidden_dim)

        # 空间注意力模块
        self.spatial_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, 1, 1),
                nn.Sigmoid()
            ) for _ in range(4)
        ])

        self.entropy_guided_attention = EntropyGuidedAttention(hidden_dim)
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.final_projection = nn.Linear(hidden_dim, 128)
        self.mi_estimator = MutualInformationEstimator(hidden_dim)

        # 特征转换层 - 从hidden_dim转换回原始维度
        self.feature_transform = nn.ModuleList([
            nn.Conv2d(hidden_dim, c, 1) for c in [64, 128, 256, 512]
        ])

    def entropy(self, x):
        p = F.softmax(x, dim=-1)
        return -torch.sum(p * torch.log(p + 1e-9), dim=-1)

    def forward(self, xA_list, memory_textA):
        batch_size = xA_list[0].shape[0]
        enhanced_features = []

        # 处理文本特征
        text_features = self.text_projection(memory_textA)  # [B, 4, H]

        # 处理每个尺度的特征
        visual_features = []
        for i, x in enumerate(xA_list[:4]):
            # 1. 视觉特征投影到高维空间
            visual_feat = self.visual_projection[i](x)  # [B, H, Hi, Wi]

            # 2. 生成文本引导的空间注意力
            text_feat = text_features[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            text_feat = text_feat.expand(-1, -1, visual_feat.size(2), visual_feat.size(3))

            combined_feat = torch.cat([visual_feat, text_feat], dim=1)
            attention_map = self.spatial_attention[i](combined_feat)

            # 3. 应用注意力和语义引导
            enhanced_feat = visual_feat * attention_map

            # 4. 转换回原始维度并加入残差连接
            output_feat = self.feature_transform[i](enhanced_feat)
            final_feat = output_feat + x

            enhanced_features.append(final_feat)

            # 收集全局特征用于后续处理
            pooled_feat = F.adaptive_avg_pool2d(enhanced_feat, (1, 1)).squeeze(-1).squeeze(-1)
            visual_features.append(pooled_feat)

        # 堆叠视觉特征
        visual_features = torch.stack(visual_features, dim=1)  # [B, 4, H]

        # 计算信息熵
        visual_entropy = self.entropy(visual_features)
        text_entropy = self.entropy(text_features)

        # 估计互信息
        mi = self.mi_estimator(visual_features.view(-1, self.hidden_dim),
                               text_features.view(-1, self.hidden_dim))

        # 使用信息熵和互信息指导注意力
        total_entropy = visual_entropy + text_entropy
        entropy_weights = F.softmax(total_entropy, dim=-1)
        mi_weight = torch.sigmoid(mi)

        # 动态融合
        attended_features = self.entropy_guided_attention(visual_features, text_features, entropy_weights)
        fused_features = mi_weight * attended_features + (1 - mi_weight) * visual_features

        # 最终特征
        final_features = self.fusion_layer(torch.cat([fused_features.mean(dim=1), text_features.mean(dim=1)], dim=-1))

        # 生成输出
        text_embeddingsA = final_features.unsqueeze(1).repeat(1, 2, 1)
        score_mapA = self.final_projection(final_features).view(batch_size, 2, 8, 8)

        return text_embeddingsA, enhanced_features

def tes_after_extract_feat_dino():
    batch_size = 16

    # 生成随机输入数据
    xA_list = [
        torch.randn(batch_size, 64, 64, 64),
        torch.randn(batch_size, 128, 32, 32),
        torch.randn(batch_size, 256, 16, 16),
        torch.randn(batch_size, 512, 8, 8),
        torch.randn(batch_size, 512, 8, 8)
    ]
    memory_textA = torch.randn(batch_size, 4, 256)

    # 创建模型
    model = AfterExtractFeatDino()

    # 前向传播
    text_embeddingsA, x_dinoA, score_mapA = model(xA_list, memory_textA)

    # 打印输出形状
    print(f"text_embeddingsA shape: {text_embeddingsA.shape}")
    print(f"x_dinoA length: {len(x_dinoA)}")
    for i, feat in enumerate(x_dinoA):
        print(f"  x_dinoA[{i}] shape: {feat.shape}")
    print(f"score_mapA shape: {score_mapA.shape}")

    # 断言检查
    assert text_embeddingsA.shape == (batch_size, 2, 1024), "text_embeddingsA 形状不正确"
    assert len(x_dinoA) == 4, "x_dinoA 长度不正确"
    assert x_dinoA[0].shape == (batch_size, 64, 64, 64), "x_dinoA[0] 形状不正确"
    assert x_dinoA[1].shape == (batch_size, 128, 32, 32), "x_dinoA[1] 形状不正确"
    assert x_dinoA[2].shape == (batch_size, 256, 16, 16), "x_dinoA[2] 形状不正确"
    assert x_dinoA[3].shape == (batch_size, 512, 8, 8), "x_dinoA[3] 形状不正确"
    assert score_mapA.shape == (batch_size, 2, 8, 8), "score_mapA 形状不正确"

    print("所有测试通过！")


def after_extract_feat_dino(xA_list, memory_textA, reference=None):
    model = AfterExtractFeatDino()
    return model(xA_list, memory_textA)


if __name__ == "__main__":
    tes_after_extract_feat_dino()