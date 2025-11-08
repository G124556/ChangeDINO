import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticConsistency(nn.Module):
    def __init__(self, logits_dim, phrase_embed_dim, hidden_dim):
        super(SemanticConsistency, self).__init__()
        self.logits_proj = nn.Linear(logits_dim, hidden_dim)
        self.phrase_proj = nn.Linear(phrase_embed_dim, hidden_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, logitsA, logitsB, phrasesA, phrasesB):
        # 投影 logits 和 phrases
        logits_featA = self.logits_proj(logitsA)
        logits_featB = self.logits_proj(logitsB)
        phrase_featA = self.phrase_proj(phrasesA)
        phrase_featB = self.phrase_proj(phrasesB)

        # 计算语义相似度
        sim_A = self.similarity(logits_featA, phrase_featA)
        sim_B = self.similarity(logits_featB, phrase_featB)

        # 计算语义变化
        semantic_change = torch.abs(sim_A - sim_B)

        # 计算语义一致性损失
        consistency_loss = F.mse_loss(semantic_change, torch.zeros_like(semantic_change))

        # 额外的语义变化预测任务
        predicted_change = self.fc(torch.abs(logits_featA - logits_featB))
        change_prediction_loss = F.mse_loss(predicted_change.squeeze(), semantic_change)

        return consistency_loss + change_prediction_loss


class MultiScaleContrastive(nn.Module):
    def __init__(self, feature_dims, output_dim=128, temperature=0.07):
        super(MultiScaleContrastive, self).__init__()
        self.projectors = nn.ModuleList([
            nn.Conv2d(dim, output_dim, 1) for dim in feature_dims
        ])
        self.temperature = temperature

    def forward(self, xA, xB):
        total_loss = 0
        for i, (proj, featA, featB) in enumerate(zip(self.projectors, xA, xB)):
            # 打印每个尺度的特征形状
            print(f"Scale {i} - FeatA shape: {featA.shape}, FeatB shape: {featB.shape}")

            # 处理特殊情况：如果featA是元组或列表，取第一个元素
            if isinstance(featA, (tuple, list)):
                featA = featA[0]
            if isinstance(featB, (tuple, list)):
                featB = featB[0]

            # 再次打印处理后的形状
            print(f"After processing - FeatA shape: {featA.shape}, FeatB shape: {featB.shape}")

            projA = F.normalize(proj(featA).flatten(2), dim=1)
            projB = F.normalize(proj(featB).flatten(2), dim=1)

            similarity = torch.matmul(projA.transpose(1, 2), projB) / self.temperature
            labels = torch.arange(similarity.size(1), device=similarity.device)
            loss = F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(1, 2), labels)
            total_loss += loss

        return total_loss / len(self.projectors)
class BoundaryAwareDetection(nn.Module):
    def __init__(self, box_dim, map_dim):
        super(BoundaryAwareDetection, self).__init__()
        self.box_encoder = nn.Sequential(
            nn.Linear(box_dim, 64),
            nn.ReLU(),
            nn.Linear(64, map_dim)
        )
        self.conv = nn.Conv2d(map_dim, map_dim, kernel_size=3, padding=1)
        self.edge_detector = nn.Conv2d(map_dim, 1, kernel_size=3, padding=1)

    def forward(self, boxesA, boxesB, change_map):
        # 编码边界框
        box_featA = self.box_encoder(boxesA).view(*change_map.shape)
        box_featB = self.box_encoder(boxesB).view(*change_map.shape)

        # 计算边界差异
        box_diff = torch.abs(box_featA - box_featB)

        # 边缘检测
        edges = self.edge_detector(self.conv(box_diff))

        # 计算边界感知损失
        boundary_loss = F.mse_loss(change_map * edges, edges)

        # IoU 损失
        iou_loss = self.iou_loss(boxesA, boxesB)

        return boundary_loss + iou_loss

    def iou_loss(self, boxesA, boxesB):
        # 计算 IoU
        intersect_mins = torch.max(boxesA[..., :2], boxesB[..., :2])
        intersect_maxes = torch.min(boxesA[..., 2:], boxesB[..., 2:])
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_mins))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        A_area = (boxesA[..., 2] - boxesA[..., 0]) * (boxesA[..., 3] - boxesA[..., 1])
        B_area = (boxesB[..., 2] - boxesB[..., 0]) * (boxesB[..., 3] - boxesB[..., 1])
        union_area = A_area + B_area - intersect_area
        iou = intersect_area / (union_area + 1e-6)

        return -torch.log(iou + 1e-6).mean()



class SpatiotemporalConsistency(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim):
        super(SpatiotemporalConsistency, self).__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.temporal_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.spatial_transformer = SpatialTransformer(hidden_dim)
        self.consistency_predictor = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, visual_feat, text_feat, change_map):
        # 特征投影
        visual = self.visual_proj(visual_feat)
        text = self.text_proj(text_feat).unsqueeze(-1).unsqueeze(-1).expand_as(visual)

        # 时间一致性
        temporal_feat = self.temporal_conv(torch.stack([visual, text, change_map.unsqueeze(1)], dim=2))
        temporal_consistency = temporal_feat.squeeze(2)

        # 空间一致性
        spatial_consistency = self.spatial_transformer(visual)

        # 预测一致性分数
        consistency_score = self.consistency_predictor(torch.cat([temporal_consistency, spatial_consistency], dim=1))

        # 计算一致性损失
        consistency_loss = F.mse_loss(consistency_score, change_map)

        return consistency_loss

class SpatialTransformer(nn.Module):
    def __init__(self, dim):
        super(SpatialTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        x = x + self.mlp(x)
        return x.permute(1, 2, 0).view(b, c, h, w)


class UncertaintyEstimation(nn.Module):
    def __init__(self, visual_dim, text_dim, box_dim, hidden_dim):
        super(UncertaintyEstimation, self).__init__()
        self.feature_fusion = FeatureFusion(visual_dim, text_dim, box_dim, hidden_dim)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1)  # 预测均值和方差
        )
        self.evidential_regression = EvidentialRegression()

    def forward(self, visual_feat, text_feat, box_feat):
        fused_feat = self.feature_fusion(visual_feat, text_feat, box_feat)
        uncertainty_params = self.uncertainty_head(fused_feat)
        return uncertainty_params


class FeatureFusion(nn.Module):
    def __init__(self, visual_dim, text_dim, box_dim, hidden_dim):
        super(FeatureFusion, self).__init__()
        self.visual_conv = nn.Conv2d(visual_dim, hidden_dim, kernel_size=1)
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.box_linear = nn.Linear(box_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

    def forward(self, visual, text, box):
        visual = self.visual_conv(visual)
        text = self.text_linear(text).unsqueeze(-1).unsqueeze(-1).expand_as(visual)
        box = self.box_linear(box).unsqueeze(-1).unsqueeze(-1).expand_as(visual)

        combined = torch.stack([visual, text, box], dim=2)
        b, c, s, h, w = combined.shape
        combined = combined.flatten(3).permute(2, 3, 0, 1)

        attn_output, _ = self.attention(combined, combined, combined)
        return attn_output.permute(2, 3, 0, 1).view(b, c, h, w)


class EvidentialRegression(nn.Module):
    def forward(self, uncertainty_params, targets):
        gamma, v, alpha, beta = uncertainty_params.chunk(4, dim=1)
        gamma = F.softplus(gamma) + 1e-6
        v = F.softplus(v) + 1e-6
        alpha = F.softplus(alpha) + 1e-6
        beta = F.softplus(beta) + 1e-6

        loss = torch.lgamma(alpha) - torch.lgamma(v / 2) - torch.lgamma((v + 1) / 2) \
               + ((v + 1) / 2) * torch.log(1 + (1 / v) * ((targets - gamma) ** 2) / beta) \
               + torch.log(beta) / 2 - alpha * torch.log(2)
        return loss.mean()


def uncertainty_guided_loss(change_map, uncertainty):
    mean, var = uncertainty.chunk(2, dim=1)
    loss = F.mse_loss(mean, change_map) * torch.exp(-var) + var
    return loss.mean()


class ChangeDetectionLoss(nn.Module):
    def __init__(self, num_classes):
        super(ChangeDetectionLoss, self).__init__()
        self.focal_loss = FocalLoss(num_classes)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, change_map, data_samples):
        gt_change = data_samples['gt_change_map']

        # Focal Loss for classification
        focal_loss = self.focal_loss(change_map, gt_change)

        # Dice Loss for region-based segmentation
        dice_loss = self.dice_loss(change_map, gt_change)

        # Boundary Loss for edge preservation
        boundary_loss = self.boundary_loss(change_map, gt_change)

        # Lovász-Softmax loss for handling imbalanced classes
        lovasz_loss = lovasz_softmax(change_map, gt_change)

        # Combine losses
        total_loss = focal_loss + dice_loss + boundary_loss + lovasz_loss

        return total_loss


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def forward(self, inputs, targets):
        smooth = 1.
        iflat = inputs.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class BoundaryLoss(nn.Module):
    def forward(self, inputs, targets):
        boundaries = F.max_pool2d(1 - targets, kernel_size=3, stride=1, padding=1) - (1 - targets)
        boundary_loss = F.binary_cross_entropy_with_logits(inputs, boundaries)
        return boundary_loss


def lovasz_softmax(inputs, targets):
    # Lovász-Softmax loss implementation
    # This is a placeholder. The actual implementation is more complex.
    return F.cross_entropy(inputs, targets)