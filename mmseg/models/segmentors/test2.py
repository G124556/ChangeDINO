import torch
import torch.nn as nn
import torch.nn.functional as F


class AfterExtractFeatDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_projection = nn.Linear(256, 1024)
        self.reference_projection = nn.Linear(4, 1024)
        self.score_map_conv = nn.Conv2d(960, 2, kernel_size=1)  # 64+128+256+512 = 960

    def forward(self, xA_list, memory_text, reference):
        # 处理 xA 列表
        x_features = xA_list[:4]  # 前4个特征图
        x_other = xA_list[4]  # 第5个特殊元素

        # 将前4个特征图调整到相同大小并连接
        target_size = x_features[0].shape[2:]  # 使用第一个特征图的大小作为目标大小
        x_resized = [F.interpolate(x, size=target_size, mode='bilinear', align_corners=False) for x in x_features]
        xA_concat = torch.cat(x_resized, dim=1)

        # 处理文本嵌入
        text_embeddingsA = self.text_projection(memory_text)  # [16, 4, 1024]
        text_embeddingsA = text_embeddingsA.mean(dim=1, keepdim=True)  # [16, 1, 1024]
        text_embeddingsA = text_embeddingsA.repeat(1, 2, 1)  # [16, 2, 1024]

        # 处理参考点
        reference_embeddingsA = self.reference_projection(reference)  # [16, 900, 1024]
        reference_embeddingsA = reference_embeddingsA[:, :2, :]  # [16, 2, 1024]
        text_embeddingsA = text_embeddingsA + reference_embeddingsA  # [16, 2, 1024]

        # 生成分数图
        score_mapA = self.score_map_conv(xA_concat)  # [16, 2, 64, 64]
        score_mapA = F.interpolate(score_mapA, size=(8, 8), mode='bilinear', align_corners=False)

        return text_embeddingsA, x_features, x_other, score_mapA


def after_extract_feat_dino(xA_list, memory_text, reference):
    model = AfterExtractFeatDino()
    return model(xA_list, memory_text, reference)



def test_after_extract_feat_dino():
    torch.manual_seed(42)

    # 生成随机输入数据
    xA_list = [
        torch.randn(16, 64, 64, 64),
        torch.randn(16, 128, 32, 32),
        torch.randn(16, 256, 16, 16),
        torch.randn(16, 512, 8, 8),
        [torch.randn(16, 2), torch.randn(16, 2)]  # 第5个元素是一个包含两个张量的列表
    ]
    memory_text = torch.randn(16, 4, 256)
    reference = torch.randn(16, 900, 4)

    # 调用函数
    text_embeddingsA, x_features, x_other, score_mapA = after_extract_feat_dino(xA_list, memory_text, reference)

    # 打印输出形状
    print(f"text_embeddingsA shape: {text_embeddingsA.shape}")
    print(f"x_features length: {len(x_features)}")
    for i, feat in enumerate(x_features):
        print(f"  x_features[{i}] shape: {feat.shape}")
    print(f"x_other length: {len(x_other)}")
    for i, tensor in enumerate(x_other):
        print(f"  x_other[{i}] shape: {tensor.shape}")
    print(f"score_mapA shape: {score_mapA.shape}")

    # 断言检查
    assert text_embeddingsA.shape == (16, 2, 1024), "text_embeddingsA 形状不正确"
    assert len(x_features) == 4, "x_features 长度不正确"
    assert all(x.shape[0] == 16 for x in x_features), "x_features batch size 不正确"
    assert len(x_other) == 2, "x_other 长度不正确"
    assert all(x.shape == (16, 2) for x in x_other), "x_other 元素形状不正确"
    assert score_mapA.shape == (16, 2, 8, 8), "score_mapA 形状不正确"

    print("所有测试通过！")

if __name__ == "__main__":
    test_after_extract_feat_dino()

