import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectCentricFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 目标区域特征增强
        self.object_enhance = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )

        # 背景区域特征处理
        self.background_process = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels)
        )

    def split_regions(self, feat, boxes):
        """根据检测框分割特征区域"""
        B, C, H, W = feat.shape
        mask = torch.zeros((B, H, W), device=feat.device)

        # 生成目标区域mask
        for b in range(B):
            for box in boxes[b]:
                x1, y1, x2, y2 = (box * torch.tensor([W, H, W, H])).int()
                mask[b, y1:y2, x1:x2] = 1

        return mask.unsqueeze(1)  # [B, 1, H, W]

    def forward(self, vis_feat, text_feat, boxes):
        """
        Args:
            vis_feat: [B, C, H, W] - 视觉特征
            text_feat: [B, C, H, W] - 文本特征
            boxes: List[Tensor] - 每张图像的检测框列表
        """
        B, C, H, W = vis_feat.shape

        # 1. 分离目标区域和背景区域
        obj_mask = self.split_regions(vis_feat, boxes)

        # 2. 特征分离
        obj_vis = vis_feat * obj_mask
        obj_text = text_feat * obj_mask
        bg_vis = vis_feat * (1 - obj_mask)
        bg_text = text_feat * (1 - obj_mask)

        # 3. 区域特征处理
        # 展平特征进行处理
        obj_vis_flat = obj_vis.flatten(2).transpose(1, 2)  # [B, HW, C]
        obj_text_flat = obj_text.flatten(2).transpose(1, 2)
        bg_vis_flat = bg_vis.flatten(2).transpose(1, 2)
        bg_text_flat = bg_text.flatten(2).transpose(1, 2)

        # 目标区域增强处理
        obj_enhanced = self.object_enhance(obj_vis_flat + obj_text_flat)

        # 背景区域处理
        bg_processed = self.background_process(bg_vis_flat + bg_text_flat)

        # 4. 特征重组
        enhanced_flat = obj_enhanced + bg_processed

        # 5. 重塑回原始维度
        enhanced = enhanced_flat.transpose(1, 2).reshape(B, C, H, W)

        return enhanced, obj_mask

class SemanticGuidanceModule(nn.Module):
    def __init__(self):
        super().__init__()

        # 特征转换层 - 将文本特征通道数调整为与视觉特征匹配
        self.text_transforms = nn.ModuleList([
            nn.Conv2d(256, 64, 1),  # 64x64
            nn.Conv2d(256, 128, 1),  # 32x32
            nn.Conv2d(256, 256, 1),  # 16x16
            nn.Conv2d(256, 512, 1)  # 8x8
        ])

        # 1. 像素级交互模块 - 关注局部像素差异
        self.pixel_interaction = nn.ModuleList([
            self._make_pixel_interaction(64),
            self._make_pixel_interaction(128),
            self._make_pixel_interaction(256),
            self._make_pixel_interaction(512)
        ])

        # 2. 区域级分析模块 - 分析局部区域变化
        self.region_analysis = nn.ModuleList([
            self._make_region_analysis(64, kernel_size=3),
            self._make_region_analysis(128, kernel_size=3),
            self._make_region_analysis(256, kernel_size=3),
            self._make_region_analysis(512, kernel_size=3)
        ])

        # 3. 语义生成网络 - 生成高级语义表示
        self.semantic_generation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048)
        )

    def _make_pixel_interaction(self, channels):
        """像素级交互模块:
        - 计算像素级的相似度矩阵
        - 基于相似度进行特征增强
        - 保留局部结构信息
        """
        return nn.Sequential(
            # 像素相似度计算
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # 局部结构增强
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # 特征重构
            nn.Conv2d(channels, channels, 1)
        )

    def _make_region_analysis(self, channels, kernel_size):
        """区域级分析模块:
        - 利用区域统计信息
        - 计算区域变化响应
        - 自适应区域聚合
        """
        return nn.ModuleDict({
            # 区域统计特征
            'stats': nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ),
            # 变化响应图
            'response': nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
                nn.Sigmoid()
            ),
            # 区域特征聚合
            'aggregate': nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
        })

    def compute_entropy(self, feat):
        """计算空间熵"""
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, -1)
        feat_norm = F.softmax(feat_flat, dim=2)
        entropy = -torch.sum(feat_norm * torch.log(feat_norm + 1e-9), dim=2)
        return entropy.unsqueeze(-1).unsqueeze(-1)  # [B, C]

    def compute_mutual_information(self, vis_feat, text_feat):
        """计算特征间的余弦相似度"""
        B, C, H, W = vis_feat.shape

        # 1. 展平空间维度
        vis_flat = vis_feat.view(B, C, -1)  # [B, C, HW]
        text_flat = text_feat.view(B, C, -1)  # [B, C, HW]

        # 2. 计算点积
        dot_product = torch.sum(vis_flat * text_flat, dim=2)  # [B, C]

        # 3. 计算L2范数
        vis_norm = torch.norm(vis_flat, p=2, dim=2)  # [B, C]
        text_norm = torch.norm(text_flat, p=2, dim=2)  # [B, C]

        # 4. 计算余弦相似度
        cosine_sim = dot_product / (vis_norm * text_norm + 1e-9)  # [B, C]

        return cosine_sim.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1]

    def feature_guidance(self, vis_feat, text_feat):
        """特征引导:
        1. 像素级交互
        2. 区域级分析
        3. 特征融合
        """
        B, C, H, W = vis_feat.shape
        if text_feat.shape[-2:] != vis_feat.shape[-2:]:
            text_feat = F.interpolate(text_feat,
                                      size=(H, W),
                                      mode='bilinear',
                                      align_corners=False)

        # 1. 像素级交互
        pixel_feat = self.pixel_interaction[0](torch.cat([vis_feat, text_feat], dim=1))

        # 2. 区域级分析
        # 2.1 计算区域统计特征
        region_stats = self.region_analysis[0]['stats'](pixel_feat)
        # 2.2 生成变化响应图
        change_response = self.region_analysis[0]['response'](
            torch.cat([region_stats, text_feat], dim=1)
        )
        # 2.3 区域特征聚合
        region_feat = self.region_analysis[0]['aggregate'](
            torch.cat([pixel_feat * change_response, region_stats], dim=1)
        )

        # 3. 计算熵和互信息
        text_entropy = self.compute_entropy(text_feat)
        mi = self.compute_mutual_information(vis_feat, text_feat)

        # 4. 加权融合
        guided_feat = region_feat * text_entropy * (1 - mi)

        return guided_feat

    def forward(self, vis_feats, text_feats):
        """
        Args:
            vis_feats: List of [B, Ci, Hi, Wi]
            text_feats: List of [B, 256, Hi, Wi]
        Returns:
            semantic_info: [B, 2, 1024]
            guided_feats: List of [B, Ci, Hi, Wi]
        """
        B = vis_feats[0].shape[0]
        guided_feats = []
        semantic_feats = []

        # 1. 多尺度特征处理
        for i, (vis_feat, text_feat) in enumerate(zip(vis_feats, text_feats)):
            # 转换文本特征
            text_transformed = self.text_transforms[i](text_feat)

            # 特征引导
            guided_feat = self.feature_guidance(vis_feat, text_transformed)
            guided_feats.append(guided_feat)

            # 收集文本特征
            semantic_feat = F.adaptive_avg_pool2d(text_feat, (16, 16))
            semantic_feats.append(semantic_feat)

        # 2. 生成高级语义信息
        semantic_features = torch.cat(semantic_feats, dim=1)  # [B, 256*4, 16, 16]
        semantic_vector = self.semantic_generation(semantic_features)  # [B, 2048]
        semantic_info = semantic_vector.view(B, 2, 1024)

        return semantic_info, guided_feats

def test_enhanced_module():
    """测试增强型语义引导模块"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建测试数据
    batch_size = 16
    vis_feats = [
        torch.randn(batch_size, 64, 64, 64).to(device),
        torch.randn(batch_size, 128, 32, 32).to(device),
        torch.randn(batch_size, 256, 16, 16).to(device),
        torch.randn(batch_size, 512, 8, 8).to(device)
    ]

    text_feats = [
        torch.randn(batch_size, 256, 32, 32).to(device),
        torch.randn(batch_size, 256, 16, 16).to(device),
        torch.randn(batch_size, 256, 8, 8).to(device),
        torch.randn(batch_size, 256, 4, 4).to(device)
    ]

    # 创建模型
    model = SemanticGuidanceModule().to(device)

    # 前向传播
    with torch.no_grad():
        semantic_info, guided_feats = model(vis_feats, text_feats)

    # 打印输出信息
    print("\n=== Output Information ===")
    print(f"Semantic info shape: {semantic_info.shape}")

    for i, feat in enumerate(guided_feats):
        print(f"\nFeature level {i}:")
        print(f"Shape: {feat.shape}")
        print(f"Value range: [{feat.min():.3f}, {feat.max():.3f}]")
        print(f"Mean: {feat.mean():.3f}")

    return semantic_info, guided_feats