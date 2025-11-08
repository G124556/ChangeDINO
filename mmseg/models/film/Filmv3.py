import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFiLMGenerator(nn.Module):
    """为每个尺度生成FiLM参数的网络"""

    def __init__(self, location_dim=256, dims=[64, 128, 256, 512]):
        super().__init__()
        self.dims = dims

        # 处理位置特征的注意力
        self.location_attention = nn.MultiheadAttention(
            embed_dim=location_dim,
            num_heads=8,
            batch_first=True
        )

        # 位置特征的投影
        self.location_proj = nn.Sequential(
            nn.Linear(location_dim, location_dim),
            nn.LayerNorm(location_dim),
            nn.ReLU()
        )

        # 为每个尺度创建参数生成器
        self.generators = nn.ModuleList([
            self._build_generator(location_dim, dim)
            for dim in dims
        ])

    def _build_generator(self, input_dim, channel_dim):
        """构建单个尺度的参数生成器"""
        return nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            # 输出gamma和beta
            nn.Linear(2048, channel_dim * channel_dim * 2 + channel_dim * channel_dim)
        )

    def forward(self, location_features):
        """
        Args:
            location_features: [B, N, D]
        Returns:
            list of (gamma_channel, beta_channel, gamma_cross) for each scale
        """
        B = location_features.shape[0]

        # 通过自注意力处理位置特征
        location_attn, _ = self.location_attention(
            location_features, location_features, location_features)

        # 全局特征
        global_feat = location_attn.mean(dim=1)  # [B, D]
        global_feat = self.location_proj(global_feat)

        # 为每个尺度生成参数
        film_params = []
        for i, generator in enumerate(self.generators):
            params = generator(global_feat)  # [B, C*C*2 + C*C]
            C = self.dims[i]

            # 分离参数
            gamma_size = C * C
            gamma_channel = params[:, :gamma_size].view(B, C, C)
            beta_channel = params[:, gamma_size:gamma_size * 2].view(B, C, C)
            gamma_cross = params[:, gamma_size * 2:].view(B, C, C)

            film_params.append((gamma_channel, beta_channel, gamma_cross))

        return film_params


class MultiScaleFiLMModulation(nn.Module):
    def __init__(self, location_dim=256, dims=[64, 128, 256, 512]):
        super().__init__()
        self.dims = dims

        # FiLM参数生成器
        self.film_generator = MultiScaleFiLMGenerator(location_dim, dims)

        # 特征投影层
        self.visual_projs = nn.ModuleList([
            nn.Conv2d(dim, dim, 1) for dim in dims
        ])

        # 特征融合权重
        self.fusion_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3)) for _ in dims
        ])
        self.softmax = nn.Softmax(dim=0)

    def apply_modulation(self, x, gamma_channel, beta_channel, gamma_cross):
        """
        应用FiLM调制
        Args:
            x: [B, C, H, W]
            gamma_channel: [B, C, C]
            beta_channel: [B, C, C]
            gamma_cross: [B, C, C]
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # [B, C, H*W]

        # 通道调制
        channel_out = torch.bmm(gamma_channel, x_flat)
        channel_out = torch.bmm(beta_channel, channel_out)

        # 交叉调制
        cross_out = torch.bmm(gamma_cross, x_flat)

        # 组合并重塑
        channel_out = channel_out.view(B, C, H, W)
        cross_out = cross_out.view(B, C, H, W)

        return channel_out, cross_out

    def forward(self, visual_features, location_features):
        """
        Args:
            visual_features: list of [B, C, H, W]
            location_features: [B, N, D]
        """
        # 生成所有尺度的FiLM参数
        film_params = self.film_generator(location_features)

        # 处理每个尺度
        outputs = []
        for i, (visual_feat, proj, weight) in enumerate(zip(
                visual_features, self.visual_projs, self.fusion_weights)):
            # 投影视觉特征
            visual_proj = proj(visual_feat)

            # 应用FiLM调制
            gamma_channel, beta_channel, gamma_cross = film_params[i]
            channel_out, cross_out = self.apply_modulation(
                visual_proj, gamma_channel, beta_channel, gamma_cross)

            # 加权融合
            weights = self.softmax(weight)
            output = (weights[0] * channel_out +
                      weights[1] * cross_out +
                      weights[2] * visual_proj)

            outputs.append(output)

        return outputs


def tet_multi_scale_film():
    # 测试数据
    batch_size = 16
    location_dim = 256
    num_queries = 1360

    # 创建输入数据
    visual_features = [
        torch.zeros(batch_size, 64, 64, 64),
        torch.zeros(batch_size, 128, 32, 32),
        torch.zeros(batch_size, 256, 16, 16),
        torch.zeros(batch_size, 512, 8, 8)
    ]
    location_features = torch.zeros(batch_size, num_queries, location_dim)

    # 初始化模型
    model = MultiScaleFiLMModulation()

    # 前向传播
    outputs = model(visual_features, location_features)

    # 验证输出
    for i, out in enumerate(outputs):
        print(f"Scale {i} output shape:", out.shape)

    return outputs


if __name__ == "__main__":
    tet_multi_scale_film()