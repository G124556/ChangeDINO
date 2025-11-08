import torch
import torch.nn as nn
import torch.nn.functional as F


class ComprehensiveChangeDetectionModel(nn.Module):
    def __init__(self):
        super(ComprehensiveChangeDetectionModel, self).__init__()
        self.visual_fusion = ImprovedVisualFusion(
            fpn_dims=[64, 128, 256, 512, 1024],
            dino_dim=256,
            output_dim=512
        )
        self.text_fusion = ImprovedTextFusion(
            input_dim=256,
            hidden_dim=512,
            output_dim=256
        )
        self.box_fusion = ImprovedBoxFusion(
            input_dim=4,
            hidden_dim=128,
            output_dim=256,
            num_references=7
        )
        self.temporal_difference = ImprovedTemporalDifference(
            visual_dim=512,
            text_dim=256,
            box_dim=256,
            hidden_dim=512,
            output_dim=256
        )
        self.attention_guided_localization = ImprovedAttentionGuidedLocalization(
            feat_dim=256,
            logits_dim=256,
            hidden_dim=512,
            output_dim=1
        )
        self.semantic_consistency = SemanticConsistency(
            logits_dim=256,
            phrase_embed_dim=768,
            hidden_dim=512
        )
        self.multi_scale_contrastive = MultiScaleContrastive(
            feature_dims=[256, 512, 1024, 2048]
        )
        self.boundary_aware_detection = BoundaryAwareDetection(
            box_dim=4,
            map_dim=128
        )
        self.spatiotemporal_consistency = SpatiotemporalConsistency(
            visual_dim=512,
            text_dim=256,
            hidden_dim=256
        )
        self.uncertainty_estimation = UncertaintyEstimation(
            visual_dim=512,
            text_dim=256,
            box_dim=128,
            hidden_dim=256
        )
        self.change_detection_loss = ChangeDetectionLoss(num_classes=2)

    def forward(self, xA, xB, outA, outB, memoryA, memoryB, memory_textA, memory_textB, referenceA, referenceB):
        # Visual fusion
        visual_feat = self.visual_fusion(xA, xB, memoryA, memoryB)

        # Text fusion
        text_feat = self.text_fusion(memory_textA, memory_textB)

        # Box fusion
        box_feat = self.box_fusion(referenceA, referenceB)

        # Temporal difference
        temp_diff = self.temporal_difference(visual_feat, text_feat, box_feat)

        # Attention guided localization
        change_map = self.attention_guided_localization(temp_diff, outA['pred_logits'], outB['pred_logits'])

        # Semantic consistency
        semantic_loss = self.semantic_consistency(outA['pred_logits'], outB['pred_logits'], memory_textA, memory_textB)

        # Multi-scale contrastive
        contrastive_loss = self.multi_scale_contrastive(xA, xB)

        # Boundary aware detection
        boundary_loss = self.boundary_aware_detection(outA['pred_boxes'], outB['pred_boxes'], change_map)

        # Spatiotemporal consistency
        consistency_loss = self.spatiotemporal_consistency(visual_feat, text_feat, change_map)

        # Uncertainty estimation
        uncertainty = self.uncertainty_estimation(visual_feat, text_feat, box_feat)

        return change_map, semantic_loss, contrastive_loss, boundary_loss, consistency_loss, uncertainty


class ImprovedVisualFusion(nn.Module):
    def __init__(self, fpn_dims, dino_dim, output_dim):
        super(ImprovedVisualFusion, self).__init__()
        self.fpn_convs = nn.ModuleList([nn.Conv2d(dim, output_dim // len(fpn_dims), kernel_size=1) for dim in fpn_dims])
        self.dino_proj = nn.Linear(dino_dim, output_dim)
        self.fusion = nn.MultiheadAttention(output_dim, 8)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, xA, xB, memoryA, memoryB):
        fpn_feats = []
        for conv, featA, featB in zip(self.fpn_convs, xA, xB):
            fpn_feats.append(F.adaptive_avg_pool2d(conv(featA + featB), (1, 1)).squeeze(-1).squeeze(-1))
        fpn_feat = torch.cat(fpn_feats, dim=1)
        dino_feat = self.dino_proj(memoryA + memoryB).mean(dim=1)
        fused_feat = self.fusion(fpn_feat.unsqueeze(0), dino_feat.unsqueeze(0), dino_feat.unsqueeze(0))[0].squeeze(0)
        return self.norm(fused_feat + fpn_feat)


class ImprovedTextFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImprovedTextFusion, self).__init__()
        self.text_proj = nn.Linear(input_dim, hidden_dim)
        self.fusion = nn.MultiheadAttention(hidden_dim, 8)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, textA, textB):
        textA = self.text_proj(textA)
        textB = self.text_proj(textB)
        fused_text = self.fusion(textA.transpose(0, 1), textB.transpose(0, 1), textB.transpose(0, 1))[0].transpose(0, 1)
        return self.norm(self.output_proj(fused_text) + textA.mean(dim=1, keepdim=True))


class ImprovedBoxFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_references):
        super(ImprovedBoxFusion, self).__init__()
        self.box_proj = nn.Linear(input_dim, hidden_dim)
        self.fusion = nn.MultiheadAttention(hidden_dim, 4)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, refA, refB):
        refA = torch.stack(refA, dim=1)
        refB = torch.stack(refB, dim=1)
        refA = self.box_proj(refA.flatten(1, 2))
        refB = self.box_proj(refB.flatten(1, 2))
        fused_ref = self.fusion(refA.transpose(0, 1), refB.transpose(0, 1), refB.transpose(0, 1))[0].transpose(0, 1)
        return self.norm(self.output_proj(fused_ref.mean(dim=1)))


class ImprovedTemporalDifference(nn.Module):
    def __init__(self, visual_dim, text_dim, box_dim, hidden_dim, output_dim):
        super(ImprovedTemporalDifference, self).__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.box_proj = nn.Linear(box_dim, hidden_dim)
        self.fusion = nn.MultiheadAttention(hidden_dim, 8)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, visual_feat, text_feat, box_feat):
        visual = self.visual_proj(visual_feat)
        text = self.text_proj(text_feat)
        box = self.box_proj(box_feat)
        combined = torch.stack([visual, text, box], dim=0)
        fused = self.fusion(combined, combined, combined)[0]
        return self.norm(self.output_proj(fused.mean(dim=0)))


class ImprovedAttentionGuidedLocalization(nn.Module):
    def __init__(self, feat_dim, logits_dim, hidden_dim, output_dim):
        super(ImprovedAttentionGuidedLocalization, self).__init__()
        self.feat_conv = nn.Conv2d(feat_dim, hidden_dim, kernel_size=3, padding=1)
        self.logits_proj = nn.Linear(logits_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 8)
        self.output_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, feat, logitsA, logitsB):
        feat = self.feat_conv(feat)
        logits_diff = self.logits_proj(logitsA - logitsB)
        b, _, h, w = feat.shape
        feat_flat = feat.flatten(2).permute(2, 0, 1)
        attended_feat = self.attention(feat_flat, logits_diff.permute(1, 0, 2), logits_diff.permute(1, 0, 2))[0]
        attended_feat = attended_feat.permute(1, 2, 0).view(b, -1, h, w)
        return self.output_conv(attended_feat)


class SemanticConsistency(nn.Module):
    def __init__(self, logits_dim, phrase_embed_dim, hidden_dim):
        super(SemanticConsistency, self).__init__()
        self.logits_proj = nn.Linear(logits_dim, hidden_dim)
        self.phrase_proj = nn.Linear(phrase_embed_dim, hidden_dim)
        self.consistency_pred = nn.Linear(hidden_dim, 1)

    def forward(self, logitsA, logitsB, phrasesA, phrasesB):
        logits_diff = self.logits_proj(logitsA - logitsB)
        phrases_diff = self.phrase_proj(phrasesA.mean(dim=1) - phrasesB.mean(dim=1))
        consistency = self.consistency_pred(logits_diff * phrases_diff.unsqueeze(1))
        return F.binary_cross_entropy_with_logits(consistency, torch.zeros_like(consistency))


class MultiScaleContrastive(nn.Module):
    def __init__(self, feature_dims, temperature=0.07):
        super(MultiScaleContrastive, self).__init__()
        self.projectors = nn.ModuleList([nn.Conv2d(dim, 128, 1) for dim in feature_dims])
        self.temperature = temperature

    def forward(self, xA, xB):
        total_loss = 0
        for proj, featA, featB in zip(self.projectors, xA, xB):
            projA = F.normalize(proj(featA).flatten(2), dim=1)
            projB = F.normalize(proj(featB).flatten(2), dim=1)
            similarity = torch.matmul(projA.transpose(1, 2), projB) / self.temperature
            labels = torch.arange(similarity.size(1), device=similarity.device)
            total_loss += F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.transpose(1, 2), labels)
        return total_loss


class BoundaryAwareDetection(nn.Module):
    def __init__(self, box_dim, map_dim):
        super(BoundaryAwareDetection, self).__init__()
        self.box_encoder = nn.Sequential(
            nn.Linear(box_dim, 64),
            nn.ReLU(),
            nn.Linear(64, map_dim)
        )
        self.edge_detector = nn.Conv2d(map_dim, 1, kernel_size=3, padding=1)

    def forward(self, boxesA, boxesB, change_map):
        box_diff = self.box_encoder(boxesA - boxesB).view(*change_map.shape)
        edges = self.edge_detector(box_diff)
        return F.mse_loss(change_map * edges, edges)


class SpatiotemporalConsistency(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim):
        super(SpatiotemporalConsistency, self).__init__()
        self.visual_conv = nn.Conv2d(visual_dim, hidden_dim, kernel_size=3, padding=1)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.consistency_pred = nn.Conv2d(hidden_dim * 2, 1, kernel_size=1)

    def forward(self, visual_feat, text_feat, change_map):
        visual = self.visual_conv(visual_feat)
        text = self.text_proj(text_feat).unsqueeze(-1).unsqueeze(-1).expand_as(visual)
        consistency = self.consistency_pred(torch.cat([visual, text], dim=1))
        return F.mse_loss(consistency, change_map)


class UncertaintyEstimation(nn.Module):
    def __init__(self, visual_dim, text_dim, box_dim, hidden_dim):
        super(UncertaintyEstimation, self).__init__()
        self.feature_fusion = nn.Sequential(
            nn.Linear(visual_dim + text_dim + box_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.uncertainty_pred = nn.Linear(hidden_dim, 1)

    def forward(self, visual_feat, text_feat, box_feat):
        combined = torch.cat([visual_feat, text_feat, box_feat], dim=1)
        fused = self.feature_fusion(combined)
        return self.uncertainty_pred(fused)


class ChangeDetectionLoss(nn.Module):
    def __init__(self, num_classes):
        super(ChangeDetectionLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return ce_loss + dice_loss


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2. * intersection + smooth) / (union + smooth)