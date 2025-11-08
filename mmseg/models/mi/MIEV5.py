import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class ObjectCentricModule(nn.Module):
    """基于目标区域的特征交互模块"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 目标区域增强
        self.object_enhance = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.LayerNorm(channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels)
        )

        # 背景区域处理
        self.background_process = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels)
        )

        # 区域交互
        self.interaction = nn.MultiheadAttention(channels, 8, dropout=0.1)

    def forward(self, vis_feat, text_feat, obj_mask):
        B, C, H, W = vis_feat.shape

        # 特征分离与处理
        obj_feat = (vis_feat * obj_mask).flatten(2).transpose(1, 2)
        bg_feat = (vis_feat * (1 - obj_mask)).flatten(2).transpose(1, 2)

        # 目标区域特征增强
        obj_enhanced = self.object_enhance(obj_feat)
        bg_processed = self.background_process(bg_feat)

        # 区域交互
        attn_out, _ = self.interaction(obj_enhanced, bg_processed, bg_processed)

        # 特征重组
        out = attn_out + obj_enhanced
        return out.transpose(1, 2).reshape(B, C, H, W)


class MultiGranularityModule(nn.Module):
    """多粒度特征交互模块"""

    def __init__(self, channels):
        super().__init__()

        # 局部特征提取
        self.local_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

        # 区域特征提取
        self.region_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.region_conv = nn.Conv2d(channels, channels, 1)

        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(channels, channels)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 局部特征
        local_feat = self.local_conv(x)

        # 区域特征
        region_feat = self.region_pool(x)
        region_feat = self.region_conv(region_feat)
        region_feat = F.interpolate(region_feat, size=x.shape[-2:], mode='bilinear')

        # 全局特征
        global_feat = self.global_pool(x)
        global_feat = self.global_fc(global_feat.squeeze(-1).squeeze(-1))
        global_feat = global_feat.view(*global_feat.shape, 1, 1).expand(-1, -1, *x.shape[-2:])

        # 特征融合
        return self.fusion(torch.cat([local_feat, region_feat, global_feat], dim=1))


class RegionAttention(nn.Module):
    """区域注意力模块"""

    def __init__(self, channels):
        super().__init__()

        # 特征转换
        self.q_conv = nn.Conv2d(channels, channels, 1)
        self.k_conv = nn.Conv2d(channels, channels, 1)
        self.v_conv = nn.Conv2d(channels, channels, 1)

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, channels, 8, 8))

        # 输出投影
        self.out_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 添加位置编码
        pos_emb = F.interpolate(self.pos_embedding, size=(H, W), mode='bilinear')
        x = x + pos_emb

        # 生成Q,K,V
        q = self.q_conv(x).flatten(2).transpose(1, 2)  # B, HW, C
        k = self.k_conv(x).flatten(2)  # B, C, HW
        v = self.v_conv(x).flatten(2).transpose(1, 2)  # B, HW, C

        # 注意力计算
        attn = F.softmax((q @ k) / (C ** 0.5), dim=-1)  # B, HW, HW
        out = (attn @ v).transpose(1, 2).reshape(B, C, H, W)  # B, C, H, W

        return self.out_proj(out)


class SemanticFusion(nn.Module):
    """语义信息融合模块"""

    def __init__(self, channels):
        super().__init__()

        # 特征变换
        self.transform = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 语义增强
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, vis_feat, text_feat):
        # 特征拼接与变换
        fused = self.transform(torch.cat([vis_feat, text_feat], dim=1))

        # 特征增强
        enhanced = self.enhance(fused)

        # 残差连接
        return fused + enhanced


class MutualInfoGuidance(nn.Module):
    """互信息引导模块"""

    def __init__(self, channels):
        super().__init__()

        # 特征变换
        self.vis_proj = nn.Linear(channels, channels)
        self.text_proj = nn.Linear(channels, channels)

        # 互信息估计器
        self.mi_estimator = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, 1)
        )

    def forward(self, vis_feat, text_feat):
        B, C, H, W = vis_feat.shape

        # 特征投影
        vis_flat = self.vis_proj(vis_feat.flatten(2).transpose(1, 2))  # B, HW, C
        text_flat = self.text_proj(text_feat.flatten(2).transpose(1, 2))  # B, HW, C

        # 计算互信息
        joint = torch.cat([vis_flat, text_flat], dim=-1)  # B, HW, 2C
        mi = self.mi_estimator(joint)  # B, HW, 1

        # 重塑为空间权重
        weights = mi.transpose(1, 2).reshape(B, 1, H, W)
        weights = torch.sigmoid(weights)

        return weights


class SemanticGuidanceModule(nn.Module):
    def __init__(self):
        super().__init__()

        # 特征转换层
        self.text_transforms = nn.ModuleList([
            nn.Conv2d(256, 64, 1),  # 64x64
            nn.Conv2d(256, 128, 1),  # 32x32
            nn.Conv2d(256, 256, 1),  # 16x16
            nn.Conv2d(256, 512, 1)  # 8x8
        ])

        # 1. 对象中心交互模块
        self.object_interaction = nn.ModuleList([
            ObjectCentricModule(64),
            ObjectCentricModule(128),
            ObjectCentricModule(256),
            ObjectCentricModule(512)
        ])

        # 2. 多粒度特征交互模块
        self.multi_granularity = nn.ModuleList([
            MultiGranularityModule(64),
            MultiGranularityModule(128),
            MultiGranularityModule(256),
            MultiGranularityModule(512)
        ])

        # 3. 区域注意力模块
        self.region_attention = nn.ModuleList([
            RegionAttention(64),
            RegionAttention(128),
            RegionAttention(256),
            RegionAttention(512)
        ])

        # 4. 语义信息融合模块
        self.semantic_fusion = nn.ModuleList([
            SemanticFusion(64),
            SemanticFusion(128),
            SemanticFusion(256),
            SemanticFusion(512)
        ])

        # 5. 互信息引导模块
        self.mi_guidance = nn.ModuleList([
            MutualInfoGuidance(64),
            MutualInfoGuidance(128),
            MutualInfoGuidance(256),
            MutualInfoGuidance(512)
        ])

        # 语义生成网络
        self.semantic_generation = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 2048)
        )

    def compute_object_mask(self, feat):
        """生成简单的目标mask用于测试"""
        B, C, H, W = feat.shape
        # 生成随机mask作为示例
        mask = torch.rand(B, 1, H, W, device=feat.device) > 0.5
        return mask.float()

    def forward(self, vis_feats, text_feats):
        """前向传播"""
        B = vis_feats[0].shape[0]
        guided_feats = []
        semantic_feats = []

        # 多尺度特征处理
        for i, (vis_feat, text_feat) in enumerate(zip(vis_feats, text_feats)):
            B, C, H, W = vis_feat.shape
            # 调整文本特征尺寸
            if text_feat.shape[-2:] != vis_feat.shape[-2:]:
                text_feat = F.interpolate(text_feat, size=(H, W), mode='bilinear', align_corners=False)
            # 转换文本特征
            text_transformed = self.text_transforms[i](text_feat)

            # 计算目标区域
            obj_mask = self.compute_object_mask(vis_feat)

            # 1. 目标中心交互
            obj_feat = self.object_interaction[i](vis_feat, text_transformed, obj_mask)

            # 2. 多粒度特征提取
            multi_feat = self.multi_granularity[i](obj_feat)

            # 3. 区域注意力
            attn_feat = self.region_attention[i](multi_feat)

            # 4. 语义融合
            semantic_feat = self.semantic_fusion[i](attn_feat, text_transformed)

            # 5. 互信息引导
            mi_weights = self.mi_guidance[i](semantic_feat, text_transformed)
            final_feat = semantic_feat * mi_weights

            guided_feats.append(final_feat)
            semantic_feats.append(F.adaptive_avg_pool2d(text_transformed, (16, 16)))

        # 生成语义信息
        semantic_features = torch.cat(semantic_feats, dim=1)  # [B, 1024, 16, 16]
        semantic_vector = self.semantic_generation(semantic_features)  # [B, 2048]
        semantic_info = semantic_vector.view(B, 2, 1024)

        return semantic_info, guided_feats


def test_semantic_guidance():
    """测试语义引导模块"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 生成测试数据
    B = 16

    # 视觉特征
    vis_feats = [
        torch.randn(B, 64, 64, 64).to(device),
        torch.randn(B, 128, 32, 32).to(device),
        torch.randn(B, 256, 16, 16).to(device),
        torch.randn(B, 512, 8, 8).to(device)
    ]

    # 文本特征
    text_feats = [
        torch.randn(B, 256, 32, 32).to(device),
        torch.randn(B, 256, 16, 16).to(device),
        torch.randn(B, 256, 8, 8).to(device),
        torch.randn(B, 256, 4, 4).to(device)
    ]

    # 创建模型
    model = SemanticGuidanceModule().to(device)

    # 前向传播
    with torch.no_grad():
        semantic_info, guided_feats = model(vis_feats, text_feats)

    # 打印信息
    print("\n=== 输出信息 ===")
    print(f"语义信息形状: {semantic_info.shape}")

    for i, feat in enumerate(guided_feats):
        print(f"\n特征层级 {i}:")
        print(f"引导后特征形状: {feat.shape}")
        print(f"数值范围: [{feat.min():.3f}, {feat.max():.3f}]")
        print(f"均值: {feat.mean():.3f}")

    # 验证维度
    assert semantic_info.shape == (B, 2, 1024), f"语义信息维度错误: {semantic_info.shape}"
    assert len(guided_feats) == 4, "特征数量错误"
    assert guided_feats[0].shape == (B, 64, 64, 64), f"特征0维度错误: {guided_feats[0].shape}"
    assert guided_feats[1].shape == (B, 128, 32, 32), f"特征1维度错误: {guided_feats[1].shape}"
    assert guided_feats[2].shape == (B, 256, 16, 16), f"特征2维度错误: {guided_feats[2].shape}"
    assert guided_feats[3].shape == (B, 512, 8, 8), f"特征3维度错误: {guided_feats[3].shape}"

    print("\n测试通过!")
    return guided_feats, semantic_info


if __name__ == "__main__":
    guided_feats, semantic_info = test_semantic_guidance()