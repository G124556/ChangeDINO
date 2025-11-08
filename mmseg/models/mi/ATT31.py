import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerSpecificAttention(nn.Module):
    def __init__(self, vision_dim, text_dim=256):
        super().__init__()
        self.text_dim = text_dim

        # 每层独立的视觉特征投影
        self.vision_proj = nn.Conv2d(vision_dim, text_dim, 1)

        # 每层独立的QKV投影
        self.q_proj = nn.Conv2d(text_dim, text_dim, 1)
        self.k_proj = nn.Conv2d(text_dim, text_dim, 1)
        self.v_proj = nn.Conv2d(text_dim, text_dim, 1)

        # 每层独立的输出投影
        self.out_proj = nn.Conv2d(text_dim, vision_dim, 1)

        # 每层独立的注意力缩放因子
        self.attention_scale = nn.Parameter(torch.FloatTensor([1.0]))


class CrossAttentionModule(nn.Module):
    def __init__(self, text_dim=256):
        super().__init__()
        self.text_dim = text_dim

        # 为每一层创建独立的注意力模块
        self.attention_layers = nn.ModuleList([
            LayerSpecificAttention(64, text_dim),  # 第一层
            LayerSpecificAttention(128, text_dim),  # 第二层
            LayerSpecificAttention(256, text_dim),  # 第三层
            LayerSpecificAttention(512, text_dim)  # 第四层
        ])

        # 为每层创建独立的语义处理器
        self.semantic_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(text_dim, text_dim, 1),
                nn.BatchNorm2d(text_dim),
                nn.ReLU()
            ) for _ in range(4)
        ])

        # 语义生成器
        self.semantic_generator = nn.Sequential(
            nn.Conv2d(text_dim * 4, text_dim, 1),
            nn.BatchNorm2d(text_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(text_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * 1024)
        )

    def process_layer(self, vision_feat, text_feat, attention_layer):
        # 1. 投影视觉特征
        v_proj_feat = attention_layer.vision_proj(vision_feat)

        # 2. 计算QKV
        q = attention_layer.q_proj(v_proj_feat)
        k = attention_layer.k_proj(text_feat)
        v = attention_layer.v_proj(text_feat)

        # 3. 计算注意力权重
        B, C, H, W = v_proj_feat.shape
        q_flat = q.flatten(2)
        k_flat = k.flatten(2)
        v_flat = v.flatten(2)

        scale = attention_layer.attention_scale * (self.text_dim ** -0.5)
        attn = torch.matmul(q_flat.transpose(1, 2), k_flat) * scale
        attn = F.softmax(attn, dim=-1)

        # 4. 应用注意力
        attended_value = torch.matmul(v_flat, attn.transpose(1, 2))
        attended_feat = attended_value.view(B, C, H, W)

        # 5. 输出投影和残差连接
        enhanced = attention_layer.out_proj(attended_feat) + vision_feat

        return enhanced

    def forward(self, vision_features, text_features):
        if isinstance(vision_features, tuple):
            vision_features = list(vision_features)
        if isinstance(text_features, tuple):
            text_features = list(text_features)

        batch_size = vision_features[0].shape[0]
        enhanced_features = []
        processed_semantic_features = []

        # 处理每一层
        for i, (v_feat, t_feat, attn_layer, semantic_proc) in enumerate(zip(
                vision_features, text_features,
                self.attention_layers, self.semantic_processors)):
            # 特征增强
            enhanced = self.process_layer(v_feat, t_feat, attn_layer)
            enhanced_features.append(enhanced)

            # 处理语义特征
            processed_semantic = semantic_proc(t_feat)
            processed_semantic_features.append(processed_semantic)

        # 生成语义信息
        smallest_size = min(feat.shape[2:] for feat in processed_semantic_features)
        aligned_features = [
            F.adaptive_avg_pool2d(feat, smallest_size)
            for feat in processed_semantic_features
        ]

        concat_features = torch.cat(aligned_features, dim=1)
        semantic_info = self.semantic_generator(concat_features)
        semantic_info = semantic_info.view(batch_size, 2, 1024)

        return semantic_info, enhanced_features


# 测试代码
def tet_cross_attention():
    # 创建模型
    model = CrossAttentionModule()
    model = model.cuda()

    # 创建测试数据
    vision_features = [
        torch.randn(16, 64, 64, 64).cuda(),
        torch.randn(16, 128, 32, 32).cuda(),
        torch.randn(16, 256, 16, 16).cuda(),
        torch.randn(16, 512, 8, 8).cuda()
    ]

    text_features = [
        torch.randn(16, 256, 32, 32).cuda(),
        torch.randn(16, 256, 16, 16).cuda(),
        torch.randn(16, 256, 8, 8).cuda(),
        torch.randn(16, 256, 4, 4).cuda()
    ]

    # 前向传播
    semantic_info, enhanced_features = model(vision_features, text_features)

    # 打印输出形状
    print("Enhanced features shapes:")
    for i, feat in enumerate(enhanced_features):
        print(f"Layer {i + 1}:", feat.shape)
    print("Semantic info shape:", semantic_info.shape)

    return enhanced_features, semantic_info


if __name__ == "__main__":
    enhanced_features, semantic_info = tet_cross_attention()