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
            output_dim=512
        )
        self.box_fusion = ImprovedBoxFusion(
            input_dim=4,
            hidden_dim=128,
            output_dim=256,
            num_references=7
        )
        self.temporal_difference = ImprovedTemporalDifference(
            visual_dim=512,
            text_dim=512,
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
            phrase_embed_dim=256,
            hidden_dim=512
        )
        self.multi_scale_contrastive = MultiScaleContrastive(
            output_dim=128,  # 可以根据需要调整
            temperature=0.07
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
        # # contrastive_loss = self.multi_scale_contrastive(xA, xB)
        #
        # # Boundary aware detection
        # boundary_loss = self.boundary_aware_detection(outA['pred_boxes'], outB['pred_boxes'], change_map)
        #
        # # Spatiotemporal consistency
        # consistency_loss = self.spatiotemporal_consistency(visual_feat, text_feat, change_map)
        #
        # # Uncertainty estimation
        # uncertainty = self.uncertainty_estimation(visual_feat, text_feat, box_feat)

        # return change_map, semantic_loss, contrastive_loss, boundary_loss, consistency_loss, uncertainty
        return change_map, semantic_loss


class ImprovedVisualFusion(nn.Module):
    def __init__(self, fpn_dims, dino_dim, output_dim):
        super(ImprovedVisualFusion, self).__init__()
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(dim, output_dim, kernel_size=1)
            for dim in fpn_dims[:-1]
        ])
        self.last_fpn_linear = nn.Linear(fpn_dims[-1], output_dim)
        self.dino_proj = nn.Linear(dino_dim, output_dim)
        self.fusion = nn.MultiheadAttention(output_dim, 8)
        self.norm = nn.LayerNorm(output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)

    def forward(self, xA, xB, memoryA, memoryB):
        fpn_feats = []
        for i, (conv, featA, featB) in enumerate(zip(self.fpn_convs, xA[:-1], xB[:-1])):
            feat = conv(featA + featB)
            fpn_feats.append(F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1))

        # 处理最后一个特殊结构
        last_featA = xA[-1][0] if isinstance(xA[-1], (list, tuple)) else xA[-1]
        last_featB = xB[-1][0] if isinstance(xB[-1], (list, tuple)) else xB[-1]
        last_feat = self.last_fpn_linear(last_featA + last_featB)
        fpn_feats.append(last_feat)

        # 堆叠 FPN 特征
        fpn_feats = torch.stack(fpn_feats, dim=1)  # (B, 5, output_dim)

        # 使用注意力机制融合 FPN 特征
        fpn_attn = self.fusion(fpn_feats.transpose(0, 1), fpn_feats.transpose(0, 1), fpn_feats.transpose(0, 1))[0]
        fpn_feat = fpn_attn.transpose(0, 1).mean(dim=1)  # (B, output_dim)

        dino_feat = self.dino_proj(memoryA.mean(dim=1) + memoryB.mean(dim=1))  # 使用平均池化

        # 融合 FPN 和 DINO 特征
        combined_feat = torch.stack([fpn_feat, dino_feat], dim=0)
        fused_feat = self.fusion(combined_feat, combined_feat, combined_feat)[0].sum(dim=0)

        fused_feat = self.norm(fused_feat)
        return self.output_proj(fused_feat)


class ImprovedTextFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImprovedTextFusion, self).__init__()
        self.text_proj = nn.Linear(input_dim, hidden_dim)
        self.fusion = nn.MultiheadAttention(hidden_dim, 8)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, textA, textB):
        # 投影文本特征
        textA = self.text_proj(textA)
        textB = self.text_proj(textB)

        # 融合文本特征
        fused_text = self.fusion(textA.transpose(0, 1), textB.transpose(0, 1), textB.transpose(0, 1))[0]
        fused_text = fused_text.transpose(0, 1)

        # 平均池化
        fused_text = fused_text.mean(dim=1)

        # 输出投影
        output = self.output_proj(fused_text)

        # 正规化
        return self.norm(output)


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
        # 打印输入特征的形状，用于调试
        print(f"Visual feat shape: {visual_feat.shape}")
        print(f"Text feat shape: {text_feat.shape}")
        print(f"Box feat shape: {box_feat.shape}")

        visual = self.visual_proj(visual_feat)
        text = self.text_proj(text_feat)
        box = self.box_proj(box_feat)

        # 确保所有特征都是 2D 张量: (batch_size, hidden_dim)
        if visual.dim() > 2:
            visual = visual.mean(dim=1)
        if text.dim() > 2:
            text = text.mean(dim=1)
        if box.dim() > 2:
            box = box.mean(dim=1)

        # 打印投影后的特征形状，用于调试
        print(f"Projected visual shape: {visual.shape}")
        print(f"Projected text shape: {text.shape}")
        print(f"Projected box shape: {box.shape}")

        # 将特征堆叠成 3D 张量: (3, batch_size, hidden_dim)
        combined = torch.stack([visual, text, box], dim=0)

        # 应用多头注意力
        fused = self.fusion(combined, combined, combined)[0]

        # 取平均得到最终特征
        fused = fused.mean(dim=0)

        output = self.output_proj(fused)
        return self.norm(output)


class ImprovedAttentionGuidedLocalization(nn.Module):
    def __init__(self, feat_dim, logits_dim, hidden_dim, output_dim):
        super(ImprovedAttentionGuidedLocalization, self).__init__()
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)
        self.logits_proj = nn.Linear(logits_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 8)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, feat, logitsA, logitsB):
        # 打印输入形状
        print(f"Feat shape: {feat.shape}")
        print(f"LogitsA shape: {logitsA.shape}")
        print(f"LogitsB shape: {logitsB.shape}")

        # 投影特征和logits
        feat = self.feat_proj(feat)  # (16, hidden_dim)
        logitsA = self.logits_proj(logitsA)  # (16, 900, hidden_dim)
        logitsB = self.logits_proj(logitsB)  # (16, 900, hidden_dim)

        # 计算logits差异
        logits_diff = logitsA - logitsB  # (16, 900, hidden_dim)

        # 将feat扩展到与logits_diff相同的形状
        feat_expanded = feat.unsqueeze(1).expand_as(logits_diff)  # (16, 900, hidden_dim)

        # 准备注意力输入
        attn_input = torch.stack([feat_expanded, logits_diff], dim=0)  # (2, 16, 900, hidden_dim)
        attn_input = attn_input.transpose(1, 2)  # (2, 900, 16, hidden_dim)

        # 应用注意力
        attended_feat, _ = self.attention(attn_input[0], attn_input[1], attn_input[1])  # (900, 16, hidden_dim)

        # 调整形状并进行最终投影
        attended_feat = attended_feat.transpose(0, 1)  # (16, 900, hidden_dim)
        change_map = self.output_proj(attended_feat)  # (16, 900, output_dim)

        return change_map.squeeze(-1)  # 返回 (16, 900) 形状的change map


class SemanticConsistency(nn.Module):
    def __init__(self, logits_dim, phrase_embed_dim, hidden_dim):
        super(SemanticConsistency, self).__init__()
        self.logits_proj = nn.Linear(logits_dim, hidden_dim)
        self.phrase_proj = nn.Linear(phrase_embed_dim, hidden_dim)
        self.consistency_pred = nn.Linear(hidden_dim, 1)

    def forward(self, logitsA, logitsB, phrasesA, phrasesB):
        # 打印输入形状
        print(f"LogitsA shape: {logitsA.shape}")
        print(f"LogitsB shape: {logitsB.shape}")
        print(f"PhrasesA shape: {phrasesA.shape}")
        print(f"PhrasesB shape: {phrasesB.shape}")

        # 处理 logits
        logits_diff = self.logits_proj(logitsA.mean(dim=1) - logitsB.mean(dim=1))  # (16, hidden_dim)

        # 处理 phrases
        phrases_diff = self.phrase_proj(phrasesA.mean(dim=1) - phrasesB.mean(dim=1))  # (16, hidden_dim)

        # 计算一致性
        consistency = self.consistency_pred(logits_diff * phrases_diff)  # (16, 1)

        return F.binary_cross_entropy_with_logits(consistency, torch.zeros_like(consistency))


class MultiScaleContrastive(nn.Module):
    def __init__(self, output_dim=128, temperature=0.07):
        super(MultiScaleContrastive, self).__init__()
        self.output_dim = output_dim
        self.temperature = temperature
        self.projectors = nn.ModuleDict()

    def forward(self, xA, xB):
        total_loss = 0
        for i, (featA, featB) in enumerate(zip(xA, xB)):


            if isinstance(featA, (tuple, list)):
                featA = featA[0]
            if isinstance(featB, (tuple, list)):
                featB = featB[0]

            print(f"After processing - FeatA shape: {featA.shape}, FeatB shape: {featB.shape}")

            in_channels = featA.shape[1]
            if f'proj_{in_channels}' not in self.projectors:
                self.projectors[f'proj_{in_channels}'] = nn.Conv2d(in_channels, self.output_dim, 1).to(featA.device)
            proj = self.projectors[f'proj_{in_channels}'].to(featA.device)

            projA = F.normalize(proj(featA).flatten(2), dim=1)
            projB = F.normalize(proj(featB).flatten(2), dim=1)

            print(f"ProjA shape: {projA.shape}, ProjB shape: {projB.shape}")

            similarity = torch.matmul(projA.transpose(1, 2), projB) / self.temperature
            print(f"Similarity shape: {similarity.shape}")

            batch_size, num_features, _ = similarity.shape
            labels = torch.arange(num_features, device=similarity.device)

            loss = F.cross_entropy(similarity.view(-1, num_features), labels.repeat(batch_size)) + \
                   F.cross_entropy(similarity.transpose(1, 2).contiguous().view(-1, num_features),
                                   labels.repeat(batch_size))

            total_loss += loss

        return total_loss / len(xA)


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