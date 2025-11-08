import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedFiLMGenerator(nn.Module):
    def __init__(self, input_dim, channel_dim, spatial_size):
        super().__init__()
        self.channel_dim = channel_dim
        self.spatial_size = spatial_size

        # 生成空间调制参数
        self.spatial_generator = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, channel_dim * spatial_size * spatial_size * 2)
        )

        # 修改通道调制参数生成器的输出维度
        self.channel_generator = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            # 为每个通道生成一个bias
            nn.Linear(1024, channel_dim * channel_dim + channel_dim)
        )

    def forward(self, x):
        B = x.shape[0]

        # 生成空间调制参数
        spatial_params = self.spatial_generator(x)
        spatial_params = spatial_params.reshape(B, 2, self.channel_dim,
                                                self.spatial_size, self.spatial_size)
        gamma_spatial, beta_spatial = spatial_params.chunk(2, dim=1)

        # 生成通道调制参数
        channel_params = self.channel_generator(x)  # [B, C*C + C]

        # 分离gamma和beta
        gamma_size = self.channel_dim * self.channel_dim
        gamma_channel = channel_params[:, :gamma_size].view(B, self.channel_dim, self.channel_dim)
        beta_channel = channel_params[:, gamma_size:].view(B, self.channel_dim)

        return (gamma_spatial.squeeze(1), beta_spatial.squeeze(1),
                gamma_channel, beta_channel)


class EnhancedFiLMModulation(nn.Module):
    def __init__(self, visual_dim, location_dim, spatial_size=64):
        super().__init__()
        self.spatial_size = spatial_size
        self.visual_dim = visual_dim

        self.film_generator = EnhancedFiLMGenerator(
            location_dim, visual_dim, spatial_size)

        self.visual_proj = nn.Conv2d(visual_dim, visual_dim, 1)

        # 用于组合不同类型的调制
        self.fusion = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(dim=0)

    def apply_spatial_modulation(self, x, gamma, beta):
        """应用空间维度的调制"""
        return gamma * x + beta

    def apply_channel_modulation(self, x, gamma, beta):
        """
        x: [B, C, H, W]
        gamma: [B, C, C]
        beta: [B, C]
        """
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)  # [B, C, H*W]

        # 通道调制
        x_modulated = torch.bmm(gamma, x_flat)  # [B, C, H*W]

        # 添加偏置，正确处理维度
        beta = beta.view(B, C, 1)
        x_modulated = x_modulated + beta

        return x_modulated.view(B, C, H, W)

    def forward(self, visual_features, location_features):
        B = visual_features.shape[0]

        # 处理位置特征
        location_mean = location_features.mean(1)

        # 生成调制参数
        gamma_spatial, beta_spatial, gamma_channel, beta_channel = \
            self.film_generator(location_mean)

        # 投影视觉特征
        visual_proj = self.visual_proj(visual_features)

        # 应用调制
        spatial_out = self.apply_spatial_modulation(
            visual_proj, gamma_spatial, beta_spatial)

        channel_out = self.apply_channel_modulation(
            visual_proj, gamma_channel, beta_channel)

        # 加权融合
        weights = self.softmax(self.fusion)
        output = (weights[0] * spatial_out +
                  weights[1] * channel_out +
                  weights[2] * visual_proj)

        return output


def tet_enhanced_film():
    # 测试数据
    batch_size = 16
    visual_channels = 64
    location_dim = 256
    height = width = 64
    num_queries = 100

    # 创建测试数据
    visual_features = torch.zeros(batch_size, visual_channels, height, width)
    location_features = torch.zeros(batch_size, num_queries, location_dim)

    # 初始化模型
    model = EnhancedFiLMModulation(visual_channels, location_dim)

    # 前向传播
    output = model(visual_features, location_features)

    # 验证维度
    print(f"Visual features: {visual_features.shape}")
    print(f"Location features: {location_features.shape}")
    print(f"Output: {output.shape}")

    return output


if __name__ == "__main__":
    tet_enhanced_film()