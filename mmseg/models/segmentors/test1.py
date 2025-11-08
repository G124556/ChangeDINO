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
        self.visual_projection = nn.ModuleList([
            nn.Conv2d(64, hidden_dim, 1),
            nn.Conv2d(128, hidden_dim, 1),
            nn.Conv2d(256, hidden_dim, 1),
            nn.Conv2d(512, hidden_dim, 1)
        ])


        # self.visual_projection = nn.ModuleList([
        #     nn.Conv2d(64*4, hidden_dim, 1),
        #     nn.Conv2d(128*4, hidden_dim, 1),
        #     nn.Conv2d(256*4, hidden_dim, 1),
        #     nn.Conv2d(512*4, hidden_dim, 1)
        # ])


        self.text_projection = nn.Linear(256, hidden_dim)
        self.entropy_guided_attention = EntropyGuidedAttention(hidden_dim)
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.final_projection = nn.Linear(hidden_dim, 128)  # 改变输出维度
        self.mi_estimator = MutualInformationEstimator(hidden_dim)

    def entropy(self, x):
        p = F.softmax(x, dim=-1)
        return -torch.sum(p * torch.log(p + 1e-9), dim=-1)

    def forward(self, xA_list, memory_textA):
        batch_size = xA_list[0].shape[0]

        # 处理视觉特征
        visual_features = []
        for i, x in enumerate(xA_list[:4]):
            x = self.visual_projection[i](x)
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (batch_size, hidden_dim)
            visual_features.append(x)
        visual_features = torch.stack(visual_features, dim=1)  # (batch_size, 4, hidden_dim)

        # 处理文本特征
        text_features = self.text_projection(memory_textA)  # (batch_size, 4, hidden_dim)

        # 计算信息熵
        visual_entropy = self.entropy(visual_features)  # (batch_size, 4)
        text_entropy = self.entropy(text_features)  # (batch_size, 4)

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
        text_embeddingsA = final_features.unsqueeze(1).repeat(1, 2, 1)  # (batch_size, 2, hidden_dim)
        x_dinoA = xA_list[:4]  # 保持原始视觉特征不变
        x_other = xA_list[4]  # 保持第5个元素不变
        score_mapA = self.final_projection(final_features).view(batch_size, 2, 8, 8)

        return text_embeddingsA, x_dinoA,score_mapA


def after_extract_feat_dino(xA_list, memory_textA, reference):
    model = AfterExtractFeatDino()
    return model(xA_list, memory_textA, reference)


# 测试函数
def test_after_extract_feat_dino():
    torch.manual_seed(42)
    batch_size = 16

    # 生成随机输入数据
    xA_list = [
        torch.randn(batch_size, 64, 64, 64),
        torch.randn(batch_size, 128, 32, 32),
        torch.randn(batch_size, 256, 16, 16),
        torch.randn(batch_size, 512, 8, 8),
        [torch.randn(batch_size, 2), torch.randn(batch_size, 2)]
    ]
    memory_textA = torch.randn(batch_size, 4, 256)
    reference = torch.randn(batch_size, 900, 4)

    # 调用函数
    text_embeddingsA, x_dinoA,  score_mapA = after_extract_feat_dino(xA_list, memory_textA, reference)

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








if __name__ == "__main__":
    test_after_extract_feat_dino()