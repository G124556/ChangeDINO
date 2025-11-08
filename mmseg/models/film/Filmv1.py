import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMGenerator(nn.Module):
    """生成FiLM参数的网络"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, output_dim * 2)  # 生成gamma和beta
        )
        self.output_dim = output_dim

    def forward(self, x):
        params = self.generator(x)
        gamma, beta = torch.chunk(params, 2, dim=1)  # 每个都是 [B, output_dim]
        return gamma, beta


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2), qkv)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.to_out(out)
        return out


class FiLMModulation(nn.Module):
    def __init__(self, visual_dim, location_dim):
        super().__init__()
        # 保持输出维度与视觉特征相同
        self.output_dim = visual_dim

        self.film_generator = FiLMGenerator(location_dim, self.output_dim)
        self.context_attention = MultiHeadAttention(location_dim)

        # 投影层保持输入输出维度相同
        self.visual_proj = nn.Conv2d(visual_dim, visual_dim, 1)

    def forward(self, visual_features, location_features):
        """
        Args:
            visual_features: [B, C, H, W]
            location_features: [B, N, D]
        """
        B, C, H, W = visual_features.shape

        # 1. 处理位置特征
        ctx = self.context_attention(location_features)  # [B, N, D]
        ctx_flat = ctx.mean(dim=1)  # [B, D]

        # 2. 生成FiLM参数
        gamma, beta = self.film_generator(ctx_flat)  # [B, C]

        # 3. 调整gamma和beta的形状以匹配视觉特征
        gamma = gamma.view(B, C, 1, 1)  # [B, C, 1, 1]
        beta = beta.view(B, C, 1, 1)  # [B, C, 1, 1]

        # 4. 投影视觉特征（维度保持不变）
        visual_proj = self.visual_proj(visual_features)  # [B, C, H, W]

        # 5. 应用FiLM调制
        modulated = gamma * visual_proj + beta  # [B, C, H, W]

        # 6. 空间注意力处理
        mod_flat = modulated.reshape(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        mod_flat = F.interpolate(mod_flat.unsqueeze(1),
                                 size=(H * W, C),
                                 mode='bilinear',
                                 align_corners=False).squeeze(1)
        out = mod_flat.permute(0, 2, 1).reshape(B, C, H, W)

        return out


def tet_film_module():
    # 测试数据维度
    batch_size = 16
    visual_channels = 64  # 视觉特征通道数
    location_dim = 256  # 位置特征维度
    height, width = 64, 64  # 特征图尺寸
    num_queries = 100  # 查询数量

    # 创建测试数据
    visual_features = torch.zeros(batch_size, visual_channels, height, width)
    location_features = torch.zeros(batch_size, num_queries, location_dim)

    # 初始化模型
    model = FiLMModulation(visual_channels, location_dim)

    # 前向传播
    output = model(visual_features, location_features)

    print("Input visual features shape:", visual_features.shape)
    print("Input location features shape:", location_features.shape)
    print("Output features shape:", output.shape)

    return output


if __name__ == "__main__":
    tet_film_module()