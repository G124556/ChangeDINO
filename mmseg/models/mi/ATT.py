import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCrossAttention(nn.Module):
    def __init__(self, text_dim=256):
        super().__init__()
        self.text_dim = text_dim

        # 视觉特征自适应投影层
        self.vision_adapter = AdaptiveProjection(out_dim=text_dim)

        # QKV投影层
        self.q_proj = nn.Conv2d(text_dim, text_dim, 1)
        self.k_proj = nn.Conv2d(text_dim, text_dim, 1)
        self.v_proj = nn.Conv2d(text_dim, text_dim, 1)

        # 输出自适应层
        self.output_adapter = AdaptiveProjection(out_dim=None)  # 动态设置输出维度

        # 高级语义信息生成器
        self.semantic_generator = SemanticGenerator(text_dim)

    def forward(self, vision_feat, text_feat):
        vision_feat=list(vision_feat)
        text_feat=list(text_feat)

        B, C, H, W = vision_feat.shape

        # 1. 将视觉特征投影到文本特征的维度
        aligned_vision = self.vision_adapter(vision_feat, out_dim=self.text_dim)

        # 2. QKV注意力计算
        q = self.q_proj(aligned_vision).flatten(2).transpose(1, 2)  # B, HW, C
        k = self.k_proj(text_feat).flatten(2)  # B, C, H'W'
        v = self.v_proj(text_feat).flatten(2)  # B, C, H'W'

        # 计算注意力分数
        attn = torch.matmul(q, k) / (self.text_dim ** 0.5)  # B, HW, H'W'
        attn = F.softmax(attn, dim=-1)

        # 应用注意力
        out = torch.matmul(attn, v.transpose(1, 2))  # B, HW, C
        out = out.transpose(1, 2).view(B, self.text_dim, H, W)

        # 3. 将特征投影回原始视觉特征维度
        enhanced = self.output_adapter(out, out_dim=C)

        # 4. 残差连接
        final_output = enhanced + vision_feat

        # 5. 生成高级语义信息
        semantic_info = self.semantic_generator(text_feat)

        return final_output, attn.view(B, H, W, -1), semantic_info


class AdaptiveProjection(nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x, out_dim=None):
        target_dim = out_dim if out_dim is not None else self.out_dim
        if target_dim is None:
            return x

        B, C, H, W = x.shape
        if C != target_dim:
            # 动态创建投影层
            proj = nn.Conv2d(C, target_dim, 1).to(x.device)
            x = proj(x)
        return x


class SemanticGenerator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # 修改降维网络结构
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, 2, 1),
            nn.BatchNorm2d(input_dim),  # 使用BatchNorm2d替代LayerNorm
            nn.ReLU(),
            nn.Conv2d(input_dim, input_dim, 3, 2, 1),
            nn.BatchNorm2d(input_dim),  # 使用BatchNorm2d替代LayerNorm
            nn.ReLU()
        )

        self.pooling = nn.AdaptiveAvgPool2d(1)

        # 添加额外的中间层以更好地处理特征
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout以防止过拟合
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2 * 1024)
        )

    def forward(self, text_feat):
        B = text_feat.shape[0]

        # 降维和池化
        x = self.conv_reduce(text_feat)
        x = self.pooling(x).squeeze(-1).squeeze(-1)  # 确保正确移除空间维度

        # 生成最终语义信息
        semantic = self.fc_layers(x)

        return semantic.view(B, 2, 1024)


def process_multi_scale_features(vision_features, text_features):
    """
    处理多尺度特征

    Args:
        vision_features: 视觉特征列表，每个元素可以有不同的空间维度
        text_features: 文本特征列表
    """
    attention_module = AdaptiveCrossAttention()
    enhanced_features = []
    attention_maps = []
    semantic_info = None

    # 对每个尺度的特征进行处理
    for i, (v_feat, t_feat) in enumerate(zip(vision_features, text_features)):
        enhanced, attn, semantic = attention_module(v_feat, t_feat)
        enhanced_features.append(enhanced)
        attention_maps.append(attn)

        # 只保留最后一个尺度的语义信息
        if i == len(vision_features) - 1:
            semantic_info = semantic

    # return enhanced_features, attention_maps, semantic_info
    return semantic_info, enhanced_features
if __name__ == "__main__":
    # 示例用法
    # 多尺度特征处理示例
    vision_features = [
        torch.randn(16, 64, 64, 64),
        torch.randn(16, 128, 32, 32),
        torch.randn(16, 256, 16, 16),
        torch.randn(16, 512, 8, 8)
    ]

    text_features = [
        torch.randn(16, 256, 32, 32),
        torch.randn(16, 256, 16, 16),
        torch.randn(16, 256, 8, 8),
        torch.randn(16, 256, 4, 4)
    ]

    guided_feats, attention_maps, semantic_info = process_multi_scale_features(
        vision_features, text_features
    )
    B=16

    assert semantic_info.shape == (B, 2, 1024), f"语义信息维度错误: {semantic_info.shape}"
    assert len(guided_feats) == 4, "特征数量错误"
    assert guided_feats[0].shape == (B, 64, 64, 64), f"特征0维度错误: {guided_feats[0].shape}"
    assert guided_feats[1].shape == (B, 128, 32, 32), f"特征1维度错误: {guided_feats[1].shape}"
    assert guided_feats[2].shape == (B, 256, 16, 16), f"特征2维度错误: {guided_feats[2].shape}"
    assert guided_feats[3].shape == (B, 512, 8, 8), f"特征3维度错误: {guided_feats[3].shape}"

    print("\n测试通过!")