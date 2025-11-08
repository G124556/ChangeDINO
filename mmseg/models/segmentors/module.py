import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualFusion(nn.Module):
    def __init__(self, resnet_dim, dino_dim, output_dim):
        super(VisualFusion, self).__init__()
        self.resnet_dim = resnet_dim
        self.dino_dim = dino_dim
        self.output_dim = output_dim

        # 多尺度特征提取
        self.resnet_conv = nn.Conv2d(resnet_dim, output_dim // 2, kernel_size=1)
        self.dino_conv = nn.Conv2d(dino_dim, output_dim // 2, kernel_size=1)

        # 跨模态注意力
        self.cross_attention = CrossModalAttention(output_dim)

        # 动态卷积核生成
        self.dynamic_conv = DynamicConvolution(output_dim)

        # 特征重校准
        self.channel_attention = ChannelAttention(output_dim)
        self.spatial_attention = SpatialAttention()

        # 残差连接
        self.residual_conv = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)

        # 归一化层
        self.norm = nn.LayerNorm(output_dim)

        # Gated Feature Fusion
        self.gate = nn.Sequential(
            nn.Conv2d(output_dim * 2, output_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, resnet_featA, resnet_featB, dino_memoryA, dino_memoryB):
        # 调整特征维度
        resnet_featA = self.resnet_conv(resnet_featA)
        resnet_featB = self.resnet_conv(resnet_featB)
        dino_memoryA = self.dino_conv(dino_memoryA.view(*resnet_featA.shape))
        dino_memoryB = self.dino_conv(dino_memoryB.view(*resnet_featB.shape))

        # 跨模态注意力融合
        fused_A = self.cross_attention(resnet_featA, dino_memoryA)
        fused_B = self.cross_attention(resnet_featB, dino_memoryB)

        # 动态卷积处理
        dynamic_A = self.dynamic_conv(fused_A)
        dynamic_B = self.dynamic_conv(fused_B)

        # 特征重校准
        recalibrated_A = self.spatial_attention(self.channel_attention(dynamic_A))
        recalibrated_B = self.spatial_attention(self.channel_attention(dynamic_B))

        # 残差连接
        residual_A = self.residual_conv(recalibrated_A)
        residual_B = self.residual_conv(recalibrated_B)

        # 最终融合
        fused = torch.cat([residual_A, residual_B], dim=1)
        gate = self.gate(fused)
        output = gate * residual_A + (1 - gate) * residual_B

        # 归一化
        output = self.norm(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return output

class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super(CrossModalAttention, self).__init__()
        self.query_conv = nn.Conv2d(dim, dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(dim, dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, resnet_feat, dino_memory):
        m_batchsize, C, height, width = resnet_feat.size()
        proj_query = self.query_conv(resnet_feat).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(dino_memory).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(dino_memory).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        return out + resnet_feat

class DynamicConvolution(nn.Module):
    def __init__(self, dim):
        super(DynamicConvolution, self).__init__()
        self.conv_generate = nn.Conv2d(dim, dim*9, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        dynamic_filters = self.conv_generate(x).view(b, c, 3, 3, h, w)
        return F.conv2d(x.view(1, -1, h, w), dynamic_filters.view(-1, 1, 3, 3), padding=1, groups=b*c).view(b, c, h, w)

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim // reduction, dim, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x * self.sigmoid(x)

class TextFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextFusion, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 自注意力机制
        self.self_attention = MultiHeadAttention(input_dim, 8)

        # 交叉注意力机制
        self.cross_attention = MultiHeadAttention(input_dim, 8)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)

        # 输出投影
        self.output_proj = nn.Linear(input_dim, output_dim)

        # 门控机制
        self.gate = nn.Linear(input_dim * 2, output_dim)

    def forward(self, memory_textA, memory_textB):
        # 自注意力
        textA = self.self_attention(memory_textA, memory_textA, memory_textA)
        textA = self.norm1(textA + memory_textA)

        textB = self.self_attention(memory_textB, memory_textB, memory_textB)
        textB = self.norm1(textB + memory_textB)

        # 交叉注意力
        textA_cross = self.cross_attention(textA, textB, textB)
        textA_cross = self.norm2(textA_cross + textA)

        textB_cross = self.cross_attention(textB, textA, textA)
        textB_cross = self.norm2(textB_cross + textB)

        # 前馈网络
        textA_out = self.ffn(textA_cross)
        textA_out = self.norm3(textA_out + textA_cross)

        textB_out = self.ffn(textB_cross)
        textB_out = self.norm3(textB_out + textB_cross)

        # 特征融合
        combined = torch.cat([textA_out, textB_out], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        fused = gate * textA_out + (1 - gate) * textB_out

        # 输出投影
        output = self.output_proj(fused)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(dim, num_heads)

    def forward(self, query, key, value):
        return self.attention(query, key, value)[0]


class BoxFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BoxFusion, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 边界框编码器
        self.box_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 时序关系建模
        self.temporal_conv = nn.Conv1d(input_dim, input_dim, kernel_size=2, padding=0)

        # 自注意力机制
        self.self_attention = MultiHeadAttention(input_dim, 4)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # 输出投影
        self.output_proj = nn.Linear(input_dim, output_dim)

        # IoU预测头
        self.iou_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, boxesA, boxesB):
        # 边界框编码
        encoded_boxesA = self.box_encoder(boxesA)
        encoded_boxesB = self.box_encoder(boxesB)

        # 时序关系建模
        temporal_features = self.temporal_conv(
            torch.stack([encoded_boxesA, encoded_boxesB], dim=2).transpose(1, 2)).transpose(1, 2)

        # 自注意力
        attended_features = self.self_attention(temporal_features, temporal_features, temporal_features)
        attended_features = self.norm1(attended_features + temporal_features)

        # 前馈网络
        output = self.ffn(attended_features)
        output = self.norm2(output + attended_features)

        # 输出投影
        fused_features = self.output_proj(output)

        # IoU预测
        iou_pred = self.iou_head(output)

        return fused_features, iou_pred


class TemporalDifference(nn.Module):
    def __init__(self, visual_dim, text_dim, box_dim, hidden_dim, output_dim):
        super(TemporalDifference, self).__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.box_dim = box_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 特征投影
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.box_proj = nn.Linear(box_dim, hidden_dim)

        # 多模态融合
        self.fusion = MultiModalFusion(hidden_dim, 3)  # 3 表示三种模态

        # 时序建模
        self.temporal_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.temporal_gru = ConvGRU(hidden_dim, hidden_dim, kernel_size=3)

        # 差异增强
        self.diff_enhance = DifferenceEnhancementModule(hidden_dim)

        # 输出投影
        self.output_proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, visual_feat, text_feat, box_feat):
        # 特征投影
        visual = self.visual_proj(visual_feat)
        text = self.text_proj(text_feat).unsqueeze(-1).unsqueeze(-1).expand_as(visual)
        box = self.box_proj(box_feat).unsqueeze(-1).unsqueeze(-1).expand_as(visual)

        # 多模态融合
        fused = self.fusion(visual, text, box)

        # 时序建模
        temp_conv = self.temporal_conv(fused)
        temp_gru, _ = self.temporal_gru(temp_conv.unsqueeze(0))
        temp_gru = temp_gru.squeeze(0)

        # 差异增强
        diff = self.diff_enhance(temp_gru)

        # 输出投影
        output = self.output_proj(diff)

        return output


class MultiModalFusion(nn.Module):
    def __init__(self, dim, num_modalities):
        super(MultiModalFusion, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, *modalities):
        # 将所有模态拼接在一起
        combined = torch.stack(modalities, dim=0)

        # 自注意力融合
        attn_output, _ = self.attention(combined, combined, combined)
        attn_output = self.norm(attn_output + combined)

        # MLP处理
        output = self.mlp(attn_output)
        output = self.norm(output + attn_output)

        return output.mean(dim=0)  # 平均池化得到最终融合结果


class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros_like(x)
        hx = torch.cat([x, h], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([x, r * h], dim=1)))
        h = (1 - z) * h + z * q
        return h, h


class DifferenceEnhancementModule(nn.Module):
    def __init__(self, dim):
        super(DifferenceEnhancementModule, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return self.gamma * out + residual

class AttentionGuidedLocalization(nn.Module):
    def __init__(self, feat_dim, logits_dim, hidden_dim, output_dim):
        super(AttentionGuidedLocalization, self).__init__()
        self.feat_dim = feat_dim
        self.logits_dim = logits_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 特征转换
        self.feat_conv = nn.Conv2d(feat_dim, hidden_dim, kernel_size=1)
        self.logits_conv = nn.Conv2d(logits_dim, hidden_dim, kernel_size=1)

        # 注意力模块
        self.attention = SpatialAttention1(hidden_dim)

        # 变化定位模块
        self.localization = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        )

        # 边缘感知模块
        self.edge_awareness = EdgeAwarenessModule(hidden_dim)

    def forward(self, temporal_diff, logits_A, logits_B):
        # 特征转换
        feat = self.feat_conv(temporal_diff)
        logits_diff = self.logits_conv(logits_B - logits_A)

        # 注意力引导
        attn_map = self.attention(feat, logits_diff)
        guided_feat = feat * attn_map

        # 边缘感知
        edge_feat = self.edge_awareness(guided_feat)

        # 变化定位
        combined = torch.cat([guided_feat, edge_feat], dim=1)
        change_map = self.localization(combined)

        return change_map

class SpatialAttention1(nn.Module):
    def __init__(self, dim):
        super(SpatialAttention1, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, feat, logits):
        combined = torch.cat([feat, logits], dim=1)
        out = F.relu(self.conv1(combined))
        attn_map = torch.sigmoid(self.conv2(out))
        return attn_map

class EdgeAwarenessModule(nn.Module):
    def __init__(self, dim):
        super(EdgeAwarenessModule, self).__init__()
        self.conv_x = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv_y = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        edge = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return edge