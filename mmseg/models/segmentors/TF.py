# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DifferenceEnhancementNetwork(nn.Module):
#     def __init__.py(self, sign_method='tanh'):
#         super().__init__.py()
#         self.conv_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(64, 256, 1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, 3, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU()
#             ),
#             nn.Sequential(
#                 nn.Conv2d(128, 256, 1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, 3, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU()
#             ),
#             nn.Sequential(
#                 nn.Conv2d(256, 256, 1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, 3, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU()
#             ),
#             nn.Sequential(
#                 nn.Conv2d(512, 256, 1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, 3, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU()
#             )
#         ])
#
#         self.diff_enhance = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(256, 256, 3, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, 3, padding=1),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU()
#             ) for _ in range(4)
#         ])
#         self.activation = nn.ReLU()
#         self.sign_method = sign_method
#
#     def smooth_sign(self, x):
#         if self.sign_method == 'tanh':
#             return torch.tanh(x * 10)  # 乘以10使函数更陡峭
#         elif self.sign_method == 'softsign':
#             return x / (1 + torch.abs(x))
#         elif self.sign_method == 'custom':
#             return x / torch.sqrt(1 + x ** 2)
#         else:
#             raise ValueError("Unknown sign method")
#
#     def nonlinear_difference(self, x, y):
#         diff = x - y
#         smooth_sign = self.smooth_sign(diff)
#         abs_diff = torch.abs(diff)
#         log_diff = torch.log1p(abs_diff)
#         exp_diff = torch.exp(torch.clamp(abs_diff, max=5)) - 1
#         enhanced_diff = smooth_sign * (log_diff + 0.1 * exp_diff)
#         return enhanced_diff
#
#     def forward(self, xdino_a, xdino_b):
#         out = []
#         for i, (feat_a, feat_b) in enumerate(zip(xdino_a, xdino_b)):
#             feat_a = self.conv_layers[i](feat_a)
#             feat_b = self.conv_layers[i](feat_b)
#             diff = self.nonlinear_difference(feat_a, feat_b)
#             enhanced_diff = self.diff_enhance[i](diff)
#             out.append(self.activation(enhanced_diff))
#         return out
#
#
# # 使用示例
#
#
#
#
# # if __name__ == "__main__":
# # # 使用示例
# #     in_channels_list = [64, 128, 256, 512]
# #     model = DifferenceEnhancementNetwork(sign_method='tanh')
# #
# #
# #     # 假设的输入
# #     x_dinoA = [torch.randn(16, c, 64 // 2 ** i, 64 // 2 ** i) for i, c in enumerate(in_channels_list)]
# #     x_dinoB = [torch.randn(16, c, 64 // 2 ** i, 64 // 2 ** i) for i, c in enumerate(in_channels_list)]
# #
# #     # 计算时频分析的差异
# #     differences = model(x_dinoA, x_dinoB)
# #
# #     # 打印输出形状
# #     for i, diff in enumerate(differences):
# #         print(f"Output {i} shape: {diff.shape}")

import torch
import torch.nn as nn
import torch.nn.functional as F


# class FiLMCrossAttentionModule(nn.Module):
#     def __init__.py(self, in_channels, text_dim, hidden_dim):
#         super().__init__.py()
#         self.visual_encoder = nn.Conv2d(in_channels, hidden_dim, 1)
#
#         # FiLM层
#         self.film_generator = nn.Sequential(
#             nn.Linear(text_dim, hidden_dim * 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim * 2, hidden_dim * 2)
#         )
#
#         # 交叉注意力
#         self.query_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.key_proj = nn.Linear(text_dim, hidden_dim)
#         self.value_proj = nn.Linear(text_dim, hidden_dim)
#         self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
#
#         # 输出层 - 修改以输出256通道
#         self.output_layer = nn.Sequential(
#             nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_dim, 256, 1)  # 改为输出256通道
#         )
#
#     def forward(self, visual_feat, text_feat):
#         # 视觉特征编码
#         visual_feat = self.visual_encoder(visual_feat)
#
#         # FiLM调制
#         film_params = self.film_generator(text_feat)
#         gamma, beta = torch.chunk(film_params, 2, dim=1)
#         gamma = gamma.unsqueeze(-1).unsqueeze(-1)
#         beta = beta.unsqueeze(-1).unsqueeze(-1)
#         film_modulated = gamma * visual_feat + beta
#
#         # 准备交叉注意力输入
#         b, c, h, w = visual_feat.shape
#         visual_feat_flat = visual_feat.flatten(2).permute(2, 0, 1)  # (h*w, b, c)
#
#         # 交叉注意力
#         queries = self.query_proj(visual_feat_flat)
#         keys = self.key_proj(text_feat).unsqueeze(0)  # (1, b, hidden_dim)
#         values = self.value_proj(text_feat).unsqueeze(0)
#         attended_feat, _ = self.attention(queries, keys, values)
#         attended_feat = attended_feat.permute(1, 2, 0).view(b, c, h, w)
#
#         # 合并FiLM和交叉注意力的结果
#         combined_feat = torch.cat([film_modulated, attended_feat], dim=1)
#
#         # 输出
#         output = self.output_layer(combined_feat)
#         return output
#
#
# class FiLMCrossAttentionChangeDetection(nn.Module):
#     def __init__.py(self, in_channels_list, text_dim, hidden_dim):
#         super().__init__.py()
#         self.film_cross_modules = nn.ModuleList([
#             FiLMCrossAttentionModule(in_channels, text_dim, hidden_dim)
#             for in_channels in in_channels_list
#         ])
#
#     def forward(self, x_dinoA, x_dinoB, text_embeddingsA, text_embeddingsB,A,B):
#         outputs = []
#         for i, (feat_a, feat_b) in enumerate(zip(x_dinoA, x_dinoB)):
#             module = self.film_cross_modules[i]
#             out_a = module(feat_a, text_embeddingsA.mean(dim=1))
#             out_b = module(feat_b, text_embeddingsB.mean(dim=1))
#             diff = torch.abs(out_a - out_b)
#             outputs.append(diff)
#         return outputs
#
#
# # 测试代码
# def test_film_cross_attention_change_detection():
#     # 设置参数
#     in_channels_list = [64, 128, 256, 512]
#     text_dim = 1024
#     hidden_dim = 256
#     batch_size = 2
#     input_size = 64
#
#     # 初始化模型
#     model = FiLMCrossAttentionChangeDetection(in_channels_list, text_dim, hidden_dim)
#
#     # 生成测试输入
#     x_dinoA = [torch.randn(batch_size, c, input_size // 2 ** i, input_size // 2 ** i) for i, c in
#                enumerate(in_channels_list)]
#     x_dinoB = [torch.randn(batch_size, c, input_size // 2 ** i, input_size // 2 ** i) for i, c in
#                enumerate(in_channels_list)]
#     text_embeddingsA = torch.randn(batch_size, 2, text_dim)
#     text_embeddingsB = torch.randn(batch_size, 2, text_dim)
#
#     # 前向传播
#     outputs = model(x_dinoA, x_dinoB, text_embeddingsA, text_embeddingsB)
#
#     # 检查输出
#     for i, out in enumerate(outputs):
#         expected_size = (batch_size, 256, input_size // 2 ** i, input_size // 2 ** i)
#         assert out.shape == expected_size, f"Output {i} shape mismatch. Expected {expected_size}, got {out.shape}"
#
#     print("All tests passed!")
#
#
# if __name__ == "__main__":
#     test_film_cross_attention_change_detection()




import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedFiLMCrossAttentionModule(nn.Module):
    def __init__(self, in_channels, text_dim, hidden_dim, num_regions=900):
        super().__init__()
        self.visual_encoder = nn.Conv2d(in_channels, hidden_dim, 1)

        # 调整文本和区域特征的维度
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.region_proj = nn.Linear(4, hidden_dim)

        # FiLM层
        self.film_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )

        # 区域特征处理
        self.region_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)

        # 交叉注意力
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 256, 1)
        )

    def forward(self, visual_feat, text_feat, region_feat):
        # 视觉特征编码
        visual_feat = self.visual_encoder(visual_feat)

        # 调整文本特征维度
        text_feat = self.text_proj(text_feat)

        # 区域特征处理
        b, num_regions, _ = region_feat.shape
        region_feat = self.region_proj(region_feat)
        region_feat = region_feat.permute(1, 0, 2)  # (num_regions, b, hidden_dim)
        region_context, _ = self.region_attention(region_feat, region_feat, region_feat)
        region_context = region_context.mean(dim=0)  # (b, hidden_dim)

        # 合并文本特征和区域上下文
        combined_context = torch.cat([text_feat, region_context], dim=1)

        # FiLM调制
        film_params = self.film_generator(combined_context)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        film_modulated = gamma * visual_feat + beta

        # 准备交叉注意力输入
        b, c, h, w = visual_feat.shape
        visual_feat_flat = visual_feat.flatten(2).permute(2, 0, 1)  # (h*w, b, c)

        # 交叉注意力
        queries = self.query_proj(visual_feat_flat)
        keys = self.key_proj(combined_context).unsqueeze(0)  # (1, b, hidden_dim)
        values = self.value_proj(combined_context).unsqueeze(0)
        attended_feat, _ = self.attention(queries, keys, values)
        attended_feat = attended_feat.permute(1, 2, 0).view(b, c, h, w)

        # 合并所有特征
        combined_feat = torch.cat([film_modulated, attended_feat, visual_feat], dim=1)

        # 输出
        output = self.output_layer(combined_feat)
        return output

# 主要的变化在 EnhancedFiLMCrossAttentionChangeDetection 类中
class FiLMCrossAttentionChangeDetection(nn.Module):
    def __init__(self, in_channels_list, text_dim, hidden_dim):
        super().__init__()
        self.film_cross_modules = nn.ModuleList([
            EnhancedFiLMCrossAttentionModule(in_channels, text_dim, hidden_dim)
            for in_channels in in_channels_list
        ])

    def forward(self, x_dinoA, x_dinoB, text_embeddingsA, text_embeddingsB, refA, refB):
        outputs = []
        for i, (feat_a, feat_b) in enumerate(zip(x_dinoA, x_dinoB)):
            module = self.film_cross_modules[i]
            out_a = module(feat_a, text_embeddingsA.mean(dim=1), refA)
            out_b = module(feat_b, text_embeddingsB.mean(dim=1), refB)
            diff = torch.abs(out_a - out_b)
            outputs.append(diff)
        return outputs

# 测试代码
def tet_enhanced_film_cross_attention_change_detection():
    # 设置参数
    in_channels_list = [64, 128, 256, 512]
    text_dim = 1024
    hidden_dim = 256
    batch_size = 2
    input_size = 64

    # 初始化模型
    model = FiLMCrossAttentionChangeDetection(in_channels_list, text_dim, hidden_dim)

    # 生成测试输入
    x_dinoA = [torch.randn(batch_size, c, input_size // 2 ** i, input_size // 2 ** i) for i, c in enumerate(in_channels_list)]
    x_dinoB = [torch.randn(batch_size, c, input_size // 2 ** i, input_size // 2 ** i) for i, c in enumerate(in_channels_list)]
    text_embeddingsA = torch.randn(batch_size, 2, text_dim)
    text_embeddingsB = torch.randn(batch_size, 2, text_dim)
    refA = torch.randn(batch_size, 900, 4)
    refB = torch.randn(batch_size, 900, 4)

    # 前向传播
    outputs = model(x_dinoA, x_dinoB, text_embeddingsA, text_embeddingsB, refA, refB)

    # 检查输出
    for i, out in enumerate(outputs):
        expected_size = (batch_size, 256, input_size // 2 ** i, input_size // 2 ** i)
        assert out.shape == expected_size, f"Output {i} shape mismatch. Expected {expected_size}, got {out.shape}"

    print("All tests passed!")

if __name__ == "__main__":
    tet_enhanced_film_cross_attention_change_detection()