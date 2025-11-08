import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from .FFPN import FlexibleFPN

# def expand_entropy_map(entropy_map, target_size):
#     """将熵图扩展到目标尺寸，保持块内值相同
#     Args:
#         entropy_map: [B, C, H, W] 例如 [16, 256, 8, 8]
#         target_size: int, 目标尺寸，例如 64
#     """
#     B, C, H, W = entropy_map.shape
#     scale_factor = target_size // H
#
#     # 1. 扩展每个值到block大小
#     expanded = entropy_map.repeat_interleave(scale_factor, dim=2). repeat_interleave(scale_factor, dim=3)
#
#     return expanded


class EntropyModule(nn.Module):
    def __init__(self, num_blocks=2, num_groups=4):
        super().__init__()
        self.num_blocks = num_blocks  # 每个维度分成几块
        self.num_groups = num_groups  # channel分组数

    def compute_grouped_block_entropy(self, feat):
        """分组计算分块熵"""
        B, C, H, W = feat.shape
        # C=[64,128,256,512]

        group_size = C // self.num_groups

        grouped_entropy_maps = []

        # 对每个group单独处理
        for i in range(self.num_groups):
            # 1. 提取当前group的特征
            group_feat = feat[:, i * group_size:(i + 1) * group_size]  # [B, C/4, H, W]

            # 2. 在通道维度上平均
            group_feat = group_feat.mean(dim=1, keepdim=True)  # [B, 1, H, W]

            # 3. 计算分块熵
            group_entropy = self.compute_block_entropy(group_feat)  # [B, 1, H_blocks, W_blocks]

            # 4. 扩展回原始大小
            expanded_entropy = F.interpolate(
                group_entropy,
                size=(H, W),
                mode='nearest'
            )  # [B, 1, H, W]

            # 5. 扩展到group的通道数
            # expanded_entropy = expanded_entropy.expand(-1, group_size, -1, -1)  # [B, C/4, H, W]
            expanded_entropy = expanded_entropy.expand(-1, 512//H, -1, -1)  # [B, C/4, H, W]

            grouped_entropy_maps.append(expanded_entropy)

        # 拼接所有group的熵图
        entropy_map = torch.cat(grouped_entropy_maps, dim=1)  # [B, C, H, W]

        # 使用1减去熵图作为权重
        return 1 - entropy_map

    # def compute_block_entropy(self, feat):
    #     """计算分块熵"""
    #     B, C, H, W = feat.shape
    #
    #     # 计算每个块的大小
    #     block_h = H // self.num_blocks
    #     block_w = W // self.num_blocks
    #
    #     # 使用adaptive_avg_pool2d进行分块
    #     blocks = F.adaptive_avg_pool2d(
    #         feat,
    #         output_size=(self.num_blocks, self.num_blocks)
    #     )
    #
    #     # 计算每个块内的熵
    #     blocks_flat = blocks.view(B, C, -1)
    #     prob = F.softmax(blocks_flat, dim=2)
    #     entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=2)
    #
    #     # 重塑为空间形状
    #     entropy = entropy.view(B, C, 1, 1)
    #
    #     # 标准化
    #     entropy_mean = entropy.mean(dim=1, keepdim=True)
    #     entropy_std = entropy.std(dim=1, keepdim=True)
    #     normalized_entropy = (entropy - entropy_mean) / (entropy_std + 1e-9)
    #
    #     return torch.sigmoid(normalized_entropy)

    def compute_block_entropy(self,feat):
        """计算分块熵
        Args:
            feat: [B, C, H, W]  比如 [16, 256, 64, 64]
        """
        B, C, H, W = feat.shape

        # 1. 将特征图等分成四块
        h_mid = H // 2
        w_mid = W // 2

        # 划分四个块
        block_1 = feat[:, :, :h_mid, :w_mid]  # 左上
        block_2 = feat[:, :, :h_mid, w_mid:]  # 右上
        block_3 = feat[:, :, h_mid:, :w_mid]  # 左下
        block_4 = feat[:, :, h_mid:, w_mid:]  # 右下

        # 2. 对每个块计算熵
        def calc_entropy(block):
            # 展平空间维度
            flat = block.reshape(B, C, -1)  # [B, C, H*W/4]
            # 计算概率
            prob = F.softmax(flat, dim=2)
            # 计算熵
            entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=2)  # [B, C]
            return entropy.mean(dim=1, keepdim=True)  # [B, 1]

        entropy_1 = calc_entropy(block_1)
        entropy_2 = calc_entropy(block_2)
        entropy_3 = calc_entropy(block_3)
        entropy_4 = calc_entropy(block_4)

        # 3. 构建2x2的熵图
        entropy_map = torch.cat([
            torch.cat([entropy_1, entropy_2], dim=1),
            torch.cat([entropy_3, entropy_4], dim=1)
        ], dim=1).view(B, 1, 2, 2)

        # 4. 标准化
        entropy_mean = entropy_map.mean()
        entropy_std = entropy_map.std()
        normalized_entropy = (entropy_map - entropy_mean) / (entropy_std + 1e-9)

        # 5. 扩展回原始尺寸
        entropy_map = F.interpolate(
            normalized_entropy,
            size=(H, W),
            mode='nearest'
        )

        # 返回1减去熵图作为权重
        return 1 - torch.sigmoid(entropy_map)



def expand_entropy_map(entropy_map, target_feat):
    """将熵图扩展到目标特征的尺寸，保持块内值相同
    Args:
        entropy_map: [B, C, H, W] 例如 [16, 256, 8, 8]
        target_feat: [B, C, H', W'] 目标特征
    """
    B, C, H, W = entropy_map.shape
    target_H = target_feat.shape[2]
    scale_factor = target_H // H

    # 确保scale_factor是整数
    scale_factor = int(scale_factor)

    # 使用nearest模式插值
    expanded = F.interpolate(
        entropy_map,
        size=(target_H, target_H),
        mode='nearest'
    )

    return expanded







class AdaptiveFeatureAttention(nn.Module):
    """自适应特征注意力模块"""

    def __init__(self, channels):
        super().__init__()

        # 局部特征增强
        self.local_enhance = nn.Sequential(
            # 深度可分离卷积捕获局部信息
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        # 特征自注意力
        self.feature_attention = FeatureSelfAttention(channels)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # 1. 局部特征增强
        local_feat = self.local_enhance(x)

        # 2. 通道注意力
        channel_weights = self.channel_attention(local_feat)
        channel_refined = local_feat * channel_weights

        # 3. 特征自注意力
        attention_feat = self.feature_attention(channel_refined)

        # 4. 输出投影
        enhanced = self.output_proj(attention_feat)

        return enhanced + x  # 残差连接


class FeatureSelfAttention(nn.Module):
    """特征自注意力模块"""

    def __init__(self, channels):
        super().__init__()
        self.scales = [1, 2, 4]  # 多尺度attention

        # 特征映射
        self.q_conv = nn.Conv2d(channels, channels, 1)
        self.k_conv = nn.Conv2d(channels, channels, 1)
        self.v_conv = nn.Conv2d(channels, channels, 1)

        # 注意力权重
        self.attention_weights = nn.Parameter(
            torch.ones(len(self.scales)) / len(self.scales)
        )

        # 输出投影
        self.output_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成Q,K,V
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)

        # 多尺度attention
        attention_maps = []

        for scale in self.scales:
            if scale > 1:
                # 下采样
                q_scaled = F.avg_pool2d(q, scale)
                k_scaled = F.avg_pool2d(k, scale)
                v_scaled = F.avg_pool2d(v, scale)
            else:
                q_scaled, k_scaled, v_scaled = q, k, v

            # 计算attention
            q_flat = q_scaled.flatten(2)  # [B, C, HW]
            k_flat = k_scaled.flatten(2)  # [B, C, HW]
            v_flat = v_scaled.flatten(2)  # [B, C, HW]

            attn = torch.bmm(q_flat.transpose(1, 2), k_flat)  # [B, HW, HW]
            attn = F.softmax(attn / (C ** 0.5), dim=-1)

            out = torch.bmm(v_flat, attn.transpose(1, 2))  # [B, C, HW]
            out = out.view(B, C, H // scale, W // scale)

            if scale > 1:
                out = F.interpolate(out, size=(H, W), mode='bilinear')

            attention_maps.append(out)

        # 加权融合不同尺度的attention map
        weights = F.softmax(self.attention_weights, dim=0)
        output = sum(w * m for w, m in zip(weights, attention_maps))

        return self.output_proj(output)


# 分块熵计算保持不变
def compute_block_entropy(feat, block_size=8):
    """计算分块熵"""
    B, C, H, W = feat.shape

    # 分块
    blocks = F.unfold(feat, kernel_size=block_size, stride=block_size)
    blocks = blocks.view(B, C, block_size * block_size, -1)

    # 计算熵
    prob = F.softmax(blocks, dim=2)
    entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=2)

    # 重塑为空间形状
    H_out, W_out = H // block_size, W // block_size
    entropy = entropy.view(B, C, H_out, W_out)

    # 标准化
    entropy_mean = entropy.mean(dim=1, keepdim=True)
    entropy_std = entropy.std(dim=1, keepdim=True)
    normalized_entropy = (entropy - entropy_mean) / (entropy_std + 1e-9)




    return expand_entropy_map(torch.sigmoid(normalized_entropy),feat)


class StructuralActivationModule(nn.Module):
    """结构激活图生成模块"""

    def __init__(self, channels, num_groups=4):
        super().__init__()
        self.num_groups = num_groups
        self.group_channels = channels // num_groups

        # 分组卷积提取特征
        self.group_extracts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.group_channels, self.group_channels, 3, padding=1, groups=self.group_channels),
                nn.BatchNorm2d(self.group_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.group_channels, self.group_channels, 1),
                nn.BatchNorm2d(self.group_channels)
            ) for _ in range(num_groups)
        ])

        # 结构响应生成
        self.response_gen = nn.Sequential(
            nn.Conv2d(self.group_channels, self.group_channels // 4, 1),
            nn.BatchNorm2d(self.group_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.group_channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 特征分组
        grouped_feats = x.chunk(self.num_groups, dim=1)  # 每组 [B, C//4, H, W]

        # 2. 分组处理和响应生成
        activation_maps = []
        for feat, extract in zip(grouped_feats, self.group_extracts):
            # 特征提取
            extracted = extract(feat)
            # 生成响应图
            response = self.response_gen(extracted)
            activation_maps.append(response)

        # 3. 拼接所有响应图
        activation_maps = torch.cat(activation_maps, dim=1)  # [B, 4, H, W]

        return activation_maps


class EnhancedFeatureModule(nn.Module):
    """增强特征融合模块"""

    def __init__(self, channels=256, num_groups=4):
        super().__init__()

        # 特征增强
        self.feature_attention = AdaptiveFeatureAttention(channels)

        # 结构激活
        self.structural_activation = StructuralActivationModule(channels, num_groups)

        # 最终融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, vis_feat, text_feat):
        # 1. 计算分块熵
        entropy_map = compute_block_entropy(text_feat)

        # 2. 文本特征增强
        enhanced_text = self.feature_attention(text_feat)

        # 3. 熵引导的特征
        entropy_guided = enhanced_text * entropy_map

        # 4. 视觉特征的结构激活
        activation_maps = self.structural_activation(vis_feat)  # [B, 4, H, W]

        # 5. 分组处理视觉特征
        B, C, H, W = vis_feat.shape
        grouped_vis = vis_feat.view(B, self.structural_activation.num_groups, -1, H, W)

        # 6. 激活图引导
        activated_vis = []
        for i in range(self.structural_activation.num_groups):
            act_feat = grouped_vis[:, i] * activation_maps[:, i:i + 1]
            activated_vis.append(act_feat)

        activated_vis = torch.cat(activated_vis, dim=1)

        # 7. 最终融合
        concatenated = torch.cat([activated_vis, entropy_guided], dim=1)
        output = self.final_fusion(concatenated)

        return output + vis_feat  # 残差连接


class MultiScaleEnhancementModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.entropy_module = EntropyModule(num_blocks=2, num_groups=4)
        # 特征转换层 - 将文本特征通道数调整为与视觉特征匹配
        self.text_transforms = nn.ModuleList([
            nn.Conv2d(256, 64, 1),  # 64x64
            nn.Conv2d(256, 128, 1),  # 32x32
            nn.Conv2d(256, 256, 1),  # 16x16
            nn.Conv2d(256, 512, 1)  # 8x8
        ])

        self.FPN1=FlexibleFPN([256,256,256,256],[256,256,256,256])
        self.FPN2=FlexibleFPN([64,128,256,512],[64,128,256,512])
        # self.FPN2=FlexibleFPN([256,256,256,256],[256,256,256,256])
        # self.FPN2=FPn()
        # 多尺度特征增强模块
        self.feature_enhancers = nn.ModuleList([
            AdaptiveFeatureAttention(64),
            AdaptiveFeatureAttention(128),
            AdaptiveFeatureAttention(256),
            AdaptiveFeatureAttention(512)
        ])

        # 高级语义生成
        self.semantic_generation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048)
        )

    def forward(self, vis_feats, text_feats):
        """
        Args:
            vis_feats: List of [B, Ci, Hi, Wi]
            text_feats: List of [B, 256, hi, wi]
        Returns:
            enhanced_feats: List of [B, Ci, Hi, Wi]
            semantic_info: [B, 2, 1024]
        """
        B = vis_feats[0].shape[0]
        enhanced_feats = []
        text_features = []




        text_feats=self.FPN1(text_feats)

        # 1. 多尺度特征处理
        for i, (vis_feat, text_feat) in enumerate(zip(vis_feats, text_feats)):
            # 转换文本特征
            text_transformed = self.text_transforms[i](text_feat)

            # 计算分块熵并扩展到目标尺寸
            # entropy_map = compute_block_entropy(text_feat)

            entropy_map = self.entropy_module.compute_grouped_block_entropy(text_feat)

            # entropy_map = F.interpolate(
            #     entropy_map,
            #     size=vis_feat.shape[-2:],
            #     mode='nearest'
            # )

            # 特征增强
            # enhanced_text = self.feature_enhancers[i](text_transformed)
            # entropy_guided = enhanced_text * entropy_map



            entropy_guided = text_transformed * entropy_map

            B, C, H, W = vis_feat.shape
            entropy_guided= F.interpolate(entropy_guided,
                          # scale_factor=2,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True)


            # 特征融合
            enhanced_feat = vis_feat + entropy_guided
            enhanced_feats.append(enhanced_feat)

            # 收集处理后的文本特征用于生成语义信息
            pooled_text = F.adaptive_avg_pool2d(text_feat, (16, 16))
            text_features.append(pooled_text)

        # 2. 生成高级语义信息

        enhanced_feats=self.FPN2(enhanced_feats)
        semantic_features = torch.cat(text_features, dim=1)  # [B, 256*4, 16, 16]
        semantic_vector = self.semantic_generation(semantic_features)  # [B, 2048]
        semantic_info = semantic_vector.view(B, 2, 1024)


        # print(text_feats[0].size(),enhanced_feats[0].size())
        # print(text_feats[1].size(),enhanced_feats[1].size())
        # print(text_feats[2].size(),enhanced_feats[2].size())
        # print(text_feats[3].size(),enhanced_feats[3].size())

        return semantic_info, enhanced_feats


def tet_multi_scale_module():
    """测试函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 生成测试数据
    B = 1

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
    model = MultiScaleEnhancementModule().to(device)

    # 前向传播
    with torch.no_grad():
        semantic_info, enhanced_feats = model(vis_feats, text_feats)

    # 打印信息
    print("\n=== Output Information ===")
    print(f"Semantic info shape: {semantic_info.shape}")

    for i, feat in enumerate(enhanced_feats):
        print(f"\nFeature level {i}:")
        print(f"Enhanced feature shape: {feat.shape}")
        print(f"Value range: [{feat.min():.3f}, {feat.max():.3f}]")
        print(f"Mean: {feat.mean():.3f}")

    # 验证维度
    assert semantic_info.shape == (B, 2, 1024), f"语义信息维度错误: {semantic_info.shape}"
    assert len(enhanced_feats) == 4, "特征数量错误"
    assert enhanced_feats[0].shape == (B, 64, 64, 64), f"特征0维度错误: {enhanced_feats[0].shape}"
    assert enhanced_feats[1].shape == (B, 128, 32, 32), f"特征1维度错误: {enhanced_feats[1].shape}"
    assert enhanced_feats[2].shape == (B, 256, 16, 16), f"特征2维度错误: {enhanced_feats[2].shape}"
    assert enhanced_feats[3].shape == (B, 512, 8, 8), f"特征3维度错误: {enhanced_feats[3].shape}"

    print("\n测试通过!")
    return semantic_info, enhanced_feats


if __name__ == "__main__":
    semantic_info, enhanced_feats = tet_multi_scale_module()