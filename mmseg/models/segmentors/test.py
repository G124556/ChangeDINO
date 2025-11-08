import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusionModule(nn.Module):
    def __init__(self, visual_dim=256, text_dim=256, hidden_dim=256):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion_layer = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, visual_feat, text_feat):
        visual_feat = self.visual_proj(visual_feat)
        text_feat = self.text_proj(text_feat)

        # Multi-head attention fusion
        fused_feat, _ = self.fusion_layer(visual_feat, text_feat, text_feat)
        fused_feat = self.layer_norm1(visual_feat + fused_feat)

        # MLP for feature enhancement
        enhanced_feat = self.mlp(fused_feat)
        enhanced_feat = self.layer_norm2(fused_feat + enhanced_feat)

        return enhanced_feat


class TemporalAttention(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.attention = nn.MultiheadAttention(input_dim, 8, batch_first=True)

    def forward(self, feat_t1, feat_t2):
        query = self.query_proj(feat_t1)
        key = self.key_proj(feat_t2)
        value = self.value_proj(feat_t2)
        attn_output, _ = self.attention(query, key, value)
        return attn_output











class ChangeLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, change_pred, fused_t1, fused_t2, temporal_diff, targets):
        # 1. 分类损失
        cls_loss = self.bce_loss(change_pred, targets)

        # 2. 对比损失（可能需要调整或移除，取决于您的具体需求）
        # 这里我们暂时移除对比损失，因为它可能不适用于像素级预测

        # 3. 时间一致性损失
        consistency_loss = self.mse_loss(fused_t1, fused_t2)

        # 组合损失
        total_loss = self.alpha * cls_loss + self.gamma * consistency_loss

        return total_loss, {
            'cls_loss': cls_loss.item(),
            'consistency_loss': consistency_loss.item()
        }


# 更新ChangeDetectionModel以包含损失计算
class ChangeDetectionModel(nn.Module):
    def __init__(self, visual_dim=256, text_dim=256, hidden_dim=256, output_size=256):
        super().__init__()
        self.output_size = output_size
        self.feature_fusion = FeatureFusionModule(visual_dim, text_dim, hidden_dim)
        self.temporal_attention = TemporalAttention(hidden_dim)
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size * output_size)
        )
        self.loss_fn = ChangeLoss()

    def forward(self, out_t1, memory_t1, memory_text_t1, reference_t1,
                out_t2, memory_t2, memory_text_t2, reference_t2, targets=None):
        # Feature fusion for each time point
        fused_t1 = self.feature_fusion(memory_t1, memory_text_t1)
        fused_t2 = self.feature_fusion(memory_t2, memory_text_t2)

        # Temporal attention
        temporal_diff = self.temporal_attention(fused_t1, fused_t2)

        # Concatenate features and predict change
        combined_feat = torch.cat([fused_t1, temporal_diff], dim=-1)  # Concatenate along feature dimension
        batch_size, seq_len, _ = combined_feat.shape

        # Apply global average pooling
        combined_feat = combined_feat.mean(dim=1)

        # Pass through final classifier
        change_pred = self.final_classifier(combined_feat)

        # Reshape to desired output size
        change_pred = change_pred.view(batch_size, self.output_size, self.output_size)

        if self.training and targets is not None:
            loss, loss_dict = self.loss_fn(change_pred, fused_t1, fused_t2, temporal_diff, targets)
            return change_pred, loss, loss_dict

        return change_pred


# 更新测试函数以包含损失计算
def test_change_detection_model():
    # 生成随机测试数据
    batch_size, seq_len, feat_dim = 16, 900, 256
    text_len = 16
    output_size = 256

    out_t1 = {
        'pred_logits': torch.randn(batch_size, seq_len, feat_dim),
        'pred_boxes': torch.randn(batch_size, seq_len, 4)
    }
    memory_t1 = torch.randn(batch_size, seq_len, feat_dim)
    memory_text_t1 = torch.randn(batch_size, text_len, feat_dim)
    reference_t1 = torch.randn(batch_size, seq_len, feat_dim)

    out_t2 = {
        'pred_logits': torch.randn(batch_size, seq_len, feat_dim),
        'pred_boxes': torch.randn(batch_size, seq_len, 4)
    }
    memory_t2 = torch.randn(batch_size, seq_len, feat_dim)
    memory_text_t2 = torch.randn(batch_size, text_len, feat_dim)
    reference_t2 = torch.randn(batch_size, seq_len, feat_dim)

    # 生成随机目标
    targets = torch.randint(0, 2, (batch_size, output_size, output_size)).float()

    # 初始化模型
    model = ChangeDetectionModel(output_size=output_size)
    model.train()  # 设置为训练模式

    # 前向传播
    change_pred, loss, loss_dict = model(out_t1, memory_t1, memory_text_t1, reference_t1,
                                         out_t2, memory_t2, memory_text_t2, reference_t2,
                                         targets)

    print("Change prediction shape:", change_pred.shape)
    print("Total loss:", loss.item())
    print("Loss components:", loss_dict)
    print("Test passed successfully!")

# # 运行测试
# test_change_detection_model()