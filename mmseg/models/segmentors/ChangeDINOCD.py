# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .module import VisualFusion
from .module import TextFusion
from .module import BoxFusion
from .module import TemporalDifference
from .module import AttentionGuidedLocalization
from .module1 import SemanticConsistency
from .module1 import MultiScaleContrastive
from .module1 import BoundaryAwareDetection
from .module1 import SpatiotemporalConsistency
from .module1 import UncertaintyEstimation
from .module1 import ChangeDetectionLoss
from .improved import ImprovedBoxFusion
from .improved import ImprovedTextFusion
from .improved import ImprovedVisualFusion
from .improved import ImprovedTemporalDifference
from .improved import ImprovedAttentionGuidedLocalization
from .module2 import ComprehensiveChangeDetectionModel
from .test import ChangeDetectionModel
from .AEG import AfterExtractFeatDino
# from .otesff import AfterExtractFeatDino
# from .test1 import AfterExtractFeatDino
# from .TF import DifferenceEnhancementNetwork
from .TF import FiLMCrossAttentionChangeDetection
from .Img_Vision import visualization_tsne
from .Vision import visualization
import cv2
from PIL import Image
import os

from ..mi.MIEV9 import MultiScaleEnhancementModule

# from ..mi.ATT2 import ParallelCrossAttention as AdaptiveCrossAttention
# from ..mi.ATT3 import CrossAttentionModule as AdaptiveCrossAttention
from ..mi.ATT4 import SemanticAwareModule as AdaptiveCrossAttention




import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_feature_heatmap(feature_map):
    """
    Generate heatmap visualization from feature map

    Args:
        feature_map: Tensor of shape (1, 256, 64, 64)

    Returns:
        numpy array: Heatmap of shape (64, 64)
    """
    # 获取特征图的激活值
    activations = feature_map.squeeze(0)  # 移除batch维度，变成(256, 64, 64)

    # 方法1：计算通道维度的平均值
    heatmap = torch.mean(activations, dim=0)

    # 或者方法2：取所有通道的最大值
    # heatmap, _ = torch.max(activations, dim=0)

    # 归一化到0-1范围
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # 转换为numpy数组
    heatmap = heatmap.cpu().detach().numpy()

    return heatmap


def plot_heatmap(heatmap, figsize=(4, 4)):
    """
    绘制热力图

    Args:
        heatmap: numpy array of shape (64, 64)
        figsize: tuple of figure size
    """
    plt.figure(figsize=figsize)

    plt.imshow(heatmap, cmap='jet')
    plt.colorbar(label='Activation Intensity')
    plt.title('Feature Map Heatmap')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 使用示例
def visualize_feature_maps(sfcs):
    # 获取第一个特征图 (1, 256, 64, 64)
    feature_map = sfcs[0]

    # 生成热力图
    heatmap = generate_feature_heatmap(feature_map)

    # 显示热力图
    plot_heatmap(heatmap)


















#
#
#
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#
from GDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
# import cv2
#
# import torch
# print("1111111111111111111111111111111")
#
model = load_model("tools/GDINO/config/GroundingDINO-SwinT-OGC.py", "tools/GDINO/weights/groundingdino_swint_ogc.pth")
# # model = load_model("tools/GDINO/config/GroundingDINO-SwinT-OGC.py", "/data2/gaoyupeng/LESPS-master/Open-GroundingDino/workdir/checkpoint0000.pth")
#
# # model = load_model("GDINO/config/GroundingDINO-SwinT-OGC.py")
# IMAGE_PATH = "/data2/gaoyupeng/LESPS-master/ALL_data/whu-cd/train/pre/1006.png"
TEXT_PROMPT1="buildings"
#
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
#
# image_source, image = load_image(IMAGE_PATH)
#
#
# # image=torch.randn(5,3,256,256)
#
# boxes, logits, phrases, out ,memory,memory_text,reference = predict(
#     model=model,
#     image=image,
#     caption=TEXT_PROMPT1,
#     box_threshold=BOX_TRESHOLD,
#     text_threshold=TEXT_TRESHOLD
# )
#
# annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
# cv2.imwrite("annotated_image1.jpg", annotated_frame)

import torch

# 假设你的输入tensor为x，尺寸为(16, 1360, 256)
def transform_features(x):
    # 1. 首先分割特征
    feat1 = x[:, :1024, :]      # (16, 1024, 256)
    feat2 = x[:, 1024:1280, :]  # (16, 256, 256)
    feat3 = x[:, 1280:1344, :]  # (16, 64, 256)
    feat4 = x[:, 1344:1360, :]  # (16, 16, 256)

    # 2. 重塑各个特征
    feat1 = feat1.reshape(-1, 32, 32, 256)    # (16, 32, 32, 256)
    feat2 = feat2.reshape(-1, 16, 16, 256)    # (16, 16, 16, 256)
    feat3 = feat3.reshape(-1, 8, 8, 256)      # (16, 8, 8, 256)
    feat4 = feat4.reshape(-1, 4, 4, 256)      # (16, 4, 4, 256)

    # 3. 调整维度顺序：从(B, H, W, C)变为(B, C, H, W)
    feat1 = feat1.permute(0, 3, 1, 2)    # (16, 256, 32, 32)
    feat2 = feat2.permute(0, 3, 1, 2)    # (16, 256, 16, 16)
    feat3 = feat3.permute(0, 3, 1, 2)    # (16, 256, 8, 8)
    feat4 = feat4.permute(0, 3, 1, 2)    # (16, 256, 4, 4)

    return [feat1, feat2, feat3, feat4]
def print_model_structure(model, indent=''):
    """
    递归打印模型的结构和层名。

    :param model: PyTorch 模型或模块
    :param indent: 用于缩进的字符串，递归调用时使用
    """
    for name, module in model.named_children():
        print(f"{indent}{name}:")

        # 如果是叶子模块（没有子模块的模块），打印其类型
        if list(module.children()) == []:
            print(f"{indent}    {module.__class__.__name__}")

        # 如果模块有参数，打印参数的形状
        if list(module.parameters()):
            for param_name, param in module.named_parameters():
                print(f"{indent}    {param_name}: {param.shape}")

        # 递归处理子模块
        print_model_structure(module, indent + '    ')








def inspect_layer_weights(model, layer_name):
    """
    检查模型中特定层的权重。

    :param model: 加载了权重的PyTorch模型
    :param layer_name: 要检查的层的名称（字符串）
    """
    for name, param in model.named_parameters():
        if layer_name in name:
            print(f"Layer: {name}")
            print(f"Shape: {param.shape}")
            print(f"Weight values:\n{param.data}")
            print(f"Mean: {param.data.mean().item()}")
            print(f"Std: {param.data.std().item()}")
            print("---")


# 使用示例
# 假设 'model' 是您已经加载了权重的模型
# inspect_layer_weights(model, 'backbone.layer1.0.conv1')

# 如果您想查看所有层的名称
def print_model_layer_names(model):
    for name, _ in model.named_parameters():
        print(name)



"""
分界线


"""

from mmseg.models.utils import resize
from mmseg.models import builder
from mmcv.cnn import ConvModule
from mmseg.models.utils.se_layer import SELayer_v2 as SELayer
from mmseg.models.utils.dino_func import dino_infer, init_dino

from ..utils.untils import tokenize

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

from ..mi.MIEV1 import SemanticGuidanceModule


import torch
import torch.nn as nn
import torch.nn.functional as F

class FConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x



# 使用示例


import torch.nn as nn

class GDINO(nn.Module):
    def __init__(self, model, pre, TEXT_PROMPT, box_threshold, text_threshold):
        super().__init__()
        self.model = model
        self.pre=pre
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.TEXT_PROMPT=TEXT_PROMPT

    def forward(self,x):
        boxes, logits, phrases ,poss,memory, memory_text  =self.model(
            model =self.pre,
            image=x,
            caption=self.TEXT_PROMPT,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold
        )

        return boxes, logits, phrases ,poss,memory, memory_text




# 在ChangeDINOCD类中






@MODELS.register_module()
class ChangeDINO(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 text_encoder: ConfigType,
                 context_decoder: ConfigType,
                 decode_head: ConfigType,
                 class_names=['remote sensing images', 'remote sensing images change area'],  #farmland change_area
                 context_length=5,
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False,
                 tau=0.07,
                 identity_head=None,
                 token_embed_dim=512, text_dim=1024,
                 # minus_channel = [256, 512, 1024, 2048],
                 # # minus_channel = [64, 128, 256, 514],
                 minus_channel = [64, 128, 256, 512],
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # self.TF1=DifferenceEnhancementNetwork(sign_method='custom')
        # print(self.TF1)

        self.TF2=FiLMCrossAttentionChangeDetection(minus_channel,text_dim,token_embed_dim//2)

        self.M1=MultiScaleEnhancementModule()
        self.M2=MultiScaleEnhancementModule()

        self.ATT1=AdaptiveCrossAttention()
        self.ATT2=AdaptiveCrossAttention()

        # print(self.TF2)
        self.SemanticGuidanceModule=SemanticGuidanceModule()

        # self.TF2=DifferenceEnhancementNetwork(sign_method='custom')
        # print(self.TF2)


        self.CD=ChangeDetectionModel()
        # self.AfterExtractFeatDino=AfterExtractFeatDino()
        self.AfterExtractFeatDino=AfterExtractFeatDino()
        print(self.AfterExtractFeatDino)

        # self.visual_fusion = ImprovedVisualFusion(
        #     # fpn_dims=[64, 128, 256, 512, 1024],
        #     fpn_dims=[64, 128, 256, 512, 1024],
        #     dino_dim=256,
        #     output_dim=512
        # )
        # self.text_fusion = ImprovedTextFusion(
        #     input_dim=256,
        #     hidden_dim=512,
        #     output_dim=256
        # )
        # self.box_fusion = ImprovedBoxFusion(
        #     input_dim=4,
        #     hidden_dim=128,
        #     output_dim=256,
        #     num_references=7
        # )
        # self.temporal_difference = ImprovedTemporalDifference(
        #     visual_dim=512,
        #     text_dim=256,
        #     box_dim=256,
        #     hidden_dim=512,
        #     output_dim=256
        # )
        # self.attention_guided_localization = ImprovedAttentionGuidedLocalization(
        #     feat_dim=256,
        #     logits_dim=256,
        #     hidden_dim=512,
        #     output_dim=1
        # )
        # self.semantic_consistency = SemanticConsistency(
        #     logits_dim=256,
        #     phrase_embed_dim=768,
        #     hidden_dim=512
        # )
        # self.multi_scale_contrastive = MultiScaleContrastive(
        #     feature_dims=[64, 128, 256, 512, 1024],  # 这应该与xA和xB中每个尺度的通道数匹配
        #     output_dim=128,  # 可以根据需要调整
        #     temperature=0.07
        # )
        # self.boundary_aware_detection = BoundaryAwareDetection(
        #     box_dim=4,
        #     map_dim=128
        # )
        # self.spatiotemporal_consistency = SpatiotemporalConsistency(
        #     visual_dim=512,
        #     text_dim=256,
        #     hidden_dim=256
        # )
        # self.uncertainty_estimation = UncertaintyEstimation(
        #     visual_dim=512,
        #     text_dim=256,
        #     box_dim=128,
        #     hidden_dim=256
        # )
        # self.change_detection_loss = ChangeDetectionLoss(num_classes=2)




        # super().__init__.py()
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            
            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                print('not DINO pre-trained weight, using DINO ViT-B-16')
                text_encoder.pretrained = '/data2/gaoyupeng/LESPS-master/ChangeDINO/pretrain/ViT-B_16.pth'
            else:
                text_encoder.pretrained = pretrained

        self.backbone = MODELS.build(backbone)


        # #
        # self.pre = load_model("/data2/gaoyupeng/LESPS-master/ChangeDINO/tools/GDINO/config/GroundingDINO-SwinT-OGC.py",
        #                       "/data2/gaoyupeng/LESPS-master/ChangeDINO/tools/GDINO/weights/groundingdino_swint_ogc.pth")
        # self.IMAGE_PATH = "/data2/gaoyupeng/LESPS-master/ALL_data/whu-cd/train/pre/1006.png"
        # self.TEXT_PROMPT = "house"
        # self.BOX_TRESHOLD = 0.35
        # self.TEXT_TRESHOLD = 0.25
        # # self.model=predict
        # self.GDINO = GDINO(model=predict,pre=load_model("/data2/gaoyupeng/LESPS-master/ChangeDINO/tools/GDINO/config/GroundingDINO-SwinT-OGC.py",
        #                       "/data2/gaoyupeng/LESPS-master/ChangeDINO/tools/GDINO/weights/groundingdino_swint_ogc.pth")
        #                       , TEXT_PROMPT=self.TEXT_PROMPT,box_threshold=self.BOX_TRESHOLD,text_threshold=self.TEXT_TRESHOLD)
        # for param in self.GDINO.parameters():
        #     param.requires_grad = False
        self.text_encoder = MODELS.build(text_encoder)
        self.context_decoder = MODELS.build(context_decoder)
        self.context_length = context_length
        self.score_concat_index = score_concat_index
        self.minus_channel = minus_channel

        assert context_feature in ['attention', 'backbone']
        self.context_feature = context_feature

        self.text_head = text_head
        self.tau = tau

        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(class_names)        

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        context_length = self.text_encoder.context_length - self.context_length
        # self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        self.contexts2 = nn.Parameter(torch.randn(1, 1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts2)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        self.FCovblock1 = FConvBlock(256,64)
        self.FCovblock2 = FConvBlock(256,128)
        self.FCovblock3 = FConvBlock(256,256)
        self.FCovblock4 = FConvBlock(256,512)


        self.minus_conv = nn.Sequential(ConvModule(
                    in_channels=self.minus_channel[0],
                    out_channels=256,
                    kernel_size=1),
                    ConvModule(
                    in_channels=self.minus_channel[1],
                    out_channels=256,
                    kernel_size=1),
                    ConvModule(
                    in_channels=self.minus_channel[2],
                    out_channels=256,
                    kernel_size=1),
                    ConvModule(
                    in_channels=self.minus_channel[3],
                    out_channels=256,
                    kernel_size=1)
                    )
        self.channel_att = nn.Sequential(SELayer(768, 256), SELayer(768, 256), SELayer(768, 256), SELayer(768, 256))

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)

        # visualize_feature_maps(x[0])


        with torch.no_grad():

            boxes, logits, phrases, out ,memory,memory_text,reference,pos = predict(
                model=model,
                image=inputs,
                caption=TEXT_PROMPT1,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
            # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)


        # print_model_structure(model)
        # inspect_layer_weights(model, 'transformer.encoder.layers.0.self_attn.value_proj.weight')

        # monitor = ParameterMonitor(model)
        #
        # monitor.print_changes()
        # monitor.update_initial_params()

        # boxes, logits, phrases ,poss,memory, memory_text  = self.GDINO(
        #     model=self.model,
        #     image=inputs,
        #     caption=self.TEXT_PROMPT,
        #     box_threshold=self.BOX_TRESHOLD,
        #     text_threshold=self.TEXT_TRESHOLD
        # )

        # boxes, logits, phrases ,poss,memory, memory_text  = self.GDINO(inputs)

        # return x
        return x, out ,memory,memory_text,reference

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        # print(inputsA.size())
        #
        # xA = self.extract_feat(inputsA)  #修改过
        # xB = self.extract_feat(inputsB)



        xA,outA ,memoryA,memory_textA,referenceA = self.extract_feat(inputsA)  #修改过
        xB,outB ,memoryB,memory_textB,referenceB = self.extract_feat(inputsB)

        # visualization(inputsA, inputsB,xA[0], batch_img_metas)

        memoryA=transform_features(memoryA)
        memoryB=transform_features(memoryB)

        # text_embeddingsA, x_dinoA,=self.SemanticGuidanceModule(xA,memoryA)
        # text_embeddingsB, x_dinoB,=self.SemanticGuidanceModule(xB,memoryB)


        # print(xA[0].size(),memoryA[0].size())

        # import pdb
        # pdb.set_trace()
        # text_embeddingsA, x_dinoA,=self.M1(xA,memoryA)
        # # text_embeddingsB, x_dinoB,=self.M2(xB,memoryB)
        # text_embeddingsB, x_dinoB,=self.M1(xB,memoryB)

        text_embeddingsA, x_dinoA,=self.ATT1(xA,memoryA)
        # text_embeddingsB, x_dinoB,=self.M2(xB,memoryB)
        text_embeddingsB, x_dinoB,=self.ATT2(xB,memoryB)




        # change_map, semantic_loss, contrastive_loss, boundary_loss, consistency_loss, uncertainty = self.CD(
        #     xA, xB, outA, outB, memoryA, memoryB, memory_textA, memory_textB, referenceA, referenceB)

        # possA[0]=self.FCovblock1(possA[0])
        # possA[1]=self.FCovblock2(possA[1])
        # possA[2]=self.FCovblock3(possA[2])
        # possA[3]=self.FCovblock4(possA[3])
        #
        # possB[0]=self.FCovblock1(possB[0])
        # possB[1]=self.FCovblock2(possB[1])
        # possB[2]=self.FCovblock3(possB[2])
        # possB[3]=self.FCovblock4(possB[3])



        x_g = xA[-1][0]+xB[-1][0]


        x_l = xA[-1][1]+xB[-1][1]



        #
        # x_cat = [torch.cat((xA[i]+possA[i], xB[i]+possB[i]), dim=1) for i in range(len(xA)-1)]
        # x_cat.append([x_g, x_l])


        # x_g = xA[-1][0]+xB[-1][0]
        #
        # x_l = xA[-1][1]+xB[-1][1]
        #
        #
        x_cat = [torch.cat((xA[i], xB[i]), dim=1) for i in range(len(xA)-1)]
        x_cat.append([x_g, x_l])



        #
        # textA, textB = self.get_cls_text(batch_img_metas, False)
        # text_embeddingsA, x_dinoA, score_mapA = self.after_extract_feat_dino(xA, textA) #1 2 1024   x_dinoA size： xA  1 2 8 8
        # text_embeddingsB, x_dinoB, score_mapB = self.after_extract_feat_dino(xB, textB)

        # text_embeddingsA, x_dinoA, score_mapA = self.after_extract_feat_dino(xA, memory_textA)
        # text_embeddingsB, x_dinoB, score_mapB = self.after_extract_feat_dino(xB, memory_textB)




        # text_embeddingsA, x_dinoA, = self.AfterExtractFeatDino(xA, memory_textA) #1 2 1024   x_dinoA size： xA  1 2 8 8
        # text_embeddingsB, x_dinoB, = self.AfterExtractFeatDino(xB, memory_textB)

        #
        # text_embeddingsA, x_dinoA, score_mapA = self.AfterExtractFeatDino(xA, memory_textA, referenceA[-1]) #1 2 1024   x_dinoA size： xA  1 2 8 8
        # text_embeddingsB, x_dinoB, score_mapB = self.AfterExtractFeatDino(xB, memory_textB, referenceB[-1])

        # visualization_tsne(x_dinoA[0], x_dinoB[0], data_samples)

        # print(batch_img_metas)
        # visualization_tsne(x_dinoA[0], x_dinoB[0], batch_img_metas)
        x_orig = [torch.cat([x_dinoA[i], x_dinoB[i]], dim=1) for i in range(len(x_dinoA))]

        # x_minus = [self.minus_conv[i](torch.abs(x_dinoA[i]-x_dinoB[i])) for i in range(len(x_dinoA))]
        # x_diff = [F.sigmoid(1-torch.cosine_similarity(x_dinoA[i], x_dinoB[i], dim=1)).unsqueeze(1) for i in range(len(x_dinoA))]
        x_minus = self.TF2(x_dinoA, x_dinoB,text_embeddingsA,text_embeddingsB,referenceA[-1],referenceB[-1])




        # visualization(inputsA, inputsB, x_minus[0], batch_img_metas)
        # visualization(inputsA, inputsB, np.abs(x_dinoA[0]-x_dinoB[0]) , batch_img_metas)
        # visualization(inputsA, inputsB, abs(x_dinoA[0]-x_dinoB[0]) , batch_img_metas)



        # x_minus = self.TF1(x_dinoA, x_dinoB)

        x_diff = [torch.sigmoid(torch.mean((x_dinoA[i] - x_dinoB[i]) ** 2, dim=1, keepdim=True)) for i in range(len(x_dinoA))]

        # score_map_diff = score_mapA-score_mapB

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()
        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        x = [self.channel_att[i](x[i]) for i in range(len(x))]

        seg_logits = self.decode_head.predict_with_text(x, text_embeddingsA, text_embeddingsB, referenceA[-1],referenceB[-1],batch_img_metas,
                                              self.test_cfg)





        return seg_logits

    # def encode_decode(self, inputs: Tensor,
    #                   batch_img_metas: List[dict]) -> Tensor:
    #     """Encode images with backbone and decode into a semantic segmentation
    #     map of the same size as input."""
    #     inputsA = inputs[:, :3, :, :]
    #     inputsB = inputs[:, 3:, :, :]
    #     xA,  = self.extract_feat(inputsA)  # 修改过
    #     xB,  = self.extract_feat(inputsB)
    #
    #     x_g = xA[-1][0] + xB[-1][0]
    #     x_l = xA[-1][1] + xB[-1][1]
    #     x_cat = [torch.cat((xA[i], xB[i]), dim=1) for i in range(len(xA) - 1)]
    #     x_cat.append([x_g, x_l])
    #
    #     textA, textB = self.get_cls_text(batch_img_metas, False)
    #
    #     text_embeddingsA, x_dinoA, score_mapA = self.after_extract_feat_dino(xA, textA)
    #     text_embeddingsB, x_dinoB, score_mapB = self.after_extract_feat_dino(xB, textB)
    #
    #     x_orig = [torch.cat([x_dinoA[i], x_dinoB[i]], dim=1) for i in range(len(x_dinoA))]
    #
    #     x_minus = [self.minus_conv[i](torch.abs(x_dinoA[i] - x_dinoB[i])) for i in range(len(x_dinoA))]
    #     x_diff = [F.sigmoid(1 - torch.cosine_similarity(x_dinoA[i], x_dinoB[i], dim=1)).unsqueeze(1) for i in
    #               range(len(x_dinoA))]
    #     score_map_diff = score_mapA - score_mapB
    #
    #     if self.with_neck:
    #         x_orig = list(self.neck(x_orig))
    #         _x_orig = x_orig
    #
    #     losses = dict()
    #     if self.text_head:
    #         x = [text_embeddingsA, ] + x_orig
    #     else:
    #         x = x_orig
    #
    #     x = [torch.cat([x[i] * x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
    #     x = [self.channel_att[i](x[i]) for i in range(len(x))]
    #
    #     seg_logits = self.decode_head.predict_with_text(x, text_embeddingsA, text_embeddingsB, batch_img_metas,
    #                                                     self.test_cfg)
    #
    #     return seg_logits








    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_train_with_text(self, x, textA, textB: List[Tensor],refA,refB,
                                            data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""



        losses = dict()
        loss_decode = self.decode_head.loss_changedino(x, textA, textB,refA,refB, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    # def _decode_head_forward_test_with_text(self, x, textA, textB, img_metas):
    #     """Run forward function and calculate loss for decode head in
    #     inference."""
    #     seg_logits = self.decode_head.forward_test_with_text(x, textA, textB, img_metas, self.test_cfg)
    #     return seg_logits

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _identity_head_forward_train(self, x, data_samples, loss_id):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_aux = self.identity_head.loss(
            x, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_aux, loss_id))
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def after_extract_feat(self, x):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map
    
    def after_extract_feat_cat(self, x):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map






    
    def after_extract_feat_dino(self, x, text):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C

        # (B, K, C)
        contexts_ = torch.cat([self.contexts2] * int(x[0].size()[0]), dim=0)
        text_embeddings = self.text_encoder(text.to(global_feat.device), contexts_).expand(B, -1, -1)
        # text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        

        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        return text_embeddings, x_orig, score_map

    # def get_cls_text(self, img_infos, train=True):
    #
    #     textA = []
    #     textB = []
    #     for i in range(len(img_infos)):
    #         print("定位")
    #         print(img_infos)
    #
    #
    #         try:
    #             foreA = ', '.join(['remote sensing image foreground objects']+img_infos[i].jsonA)
    #             foreB = ', '.join(['remote sensing image foreground objects']+img_infos[i].jsonB)
    #         except:
    #             foreA = ', '.join(['remote sensing image foreground objects']+img_infos[i]['jsonA'])
    #             foreB = ', '.join(['remote sensing image foreground objects']+img_infos[i]['jsonB'])
    #         backA = ', '.join(['remote sensing image background objects'])
    #         backB = ', '.join(['remote sensing image background objects'])
    #
    #         textA.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backA, foreA]]).unsqueeze(0))
    #         textB.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backB, foreB]]).unsqueeze(0))
    #     return torch.cat(textA, dim=0), torch.cat(textB, dim=0)

    def get_cls_text(self, img_infos, train=True):
        textA = []
        textB = []
        for i in range(len(img_infos)):
            # print("Processing sample:", i)
            # sample = img_infos[i]
            #
            # # 打印样本的所有属性，以了解可用的信息
            # print("Sample attributes:", dir(sample))
            #
            # # 尝试访问元信息
            # if hasattr(sample, 'metainfo'):
            #     print("Metainfo:", sample.metainfo)

            # 由于我们不确定jsonA和jsonB在哪里，我们暂时使用默认值
            foreA = 'remote sensing image foreground objects'
            foreB = 'remote sensing image foreground objects'
            backA = 'remote sensing image background objects'
            backB = 'remote sensing image background objects'

            textA.append(
                torch.cat([tokenize(c, context_length=self.context_length) for c in [backA, foreA]]).unsqueeze(0))
            textB.append(
                torch.cat([tokenize(c, context_length=self.context_length) for c in [backB, foreB]]).unsqueeze(0))

        return torch.cat(textA, dim=0), torch.cat(textB, dim=0)

    def loss1(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        inputsA = inputs[:, :3, :, :]  # inputsA :(1,3,256,256)
        inputsB = inputs[:, 3:, :, :]  # inputsB :(1,3,256,256)
        # xA = self.extract_feat(inputsA)  #resnet：1 64 64 64   1024  8 8
        # xB = self.extract_feat(inputsB)

        xA, boxes, logits, phrases, out, memory, memory_text, reference = self.extract_feat(inputsA)  # 修改过
        xB, boxes, logits, phrases, out, memory, memory_text, reference = self.extract_feat(inputsB)

        # for name, param in self.extract_feat.named_parameters():
        #     if param.requires_grad:
        #         print(f"Name: {name}, Shape: {param.shape}")

        #
        # possA[0]=self.FCovblock1(possA[0])
        # possA[1]=self.FCovblock2(possA[1])
        # possA[2]=self.FCovblock3(possA[2])
        # possA[3]=self.FCovblock4(possA[3])
        #
        # possB[0]=self.FCovblock1(possB[0])
        # possB[1]=self.FCovblock2(possB[1])
        # possB[2]=self.FCovblock3(possB[2])
        # possB[3]=self.FCovblock4(possB[3])

        x_g = xA[-1][0] + xB[-1][0]

        x_l = xA[-1][1] + xB[-1][1]

        # x_cat = [torch.cat((xA[i]+possA[i], xB[i]+possB[i]), dim=1) for i in range(len(xA)-1)]
        x_cat = [torch.cat((xA[i], xB[i]), dim=1) for i in range(len(xA) - 1)]
        x_cat.append([x_g, x_l])

        textA, textB = self.get_cls_text(data_samples)  # 1 2 64
        text_embeddingsA, x_dinoA, score_mapA = self.after_extract_feat_dino(xA,
                                                                             textA)  # 1 2 1024   x_dinoA size： xA  1 2 8 8
        text_embeddingsB, x_dinoB, score_mapB = self.after_extract_feat_dino(xB, textB)

        # text_embeddingsA, x_dinoA, score_mapA = self.after_extract_feat_dino(xA, memory_textA) #1 2 1024   x_dinoA size： xA  1 2 8 8
        # text_embeddingsB, x_dinoB, score_mapB = self.after_extract_feat_dino(xB, memory_textB)

        x_orig = [torch.cat([x_dinoA[i], x_dinoB[i]], dim=1) for i in range(len(x_dinoA))]  # 2* xA







        x_minus = [self.minus_conv[i](torch.abs(x_dinoA[i] - x_dinoB[i])) for i in range(len(x_dinoA))]
        x_diff = [F.sigmoid(1 - torch.cosine_similarity(x_dinoA[i], x_dinoB[i], dim=1)).unsqueeze(1) for i in
                  range(len(x_dinoA))]
        # x_diff = [torch.sigmoid(torch.mean((x_dinoA[i] - x_dinoB[i]) ** 2, dim=1, keepdim=True)) for i in
        #           range(len(x_dinoA))]


        score_map_diff = score_mapA - score_mapB

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()
        if self.text_head:
            x = [text_embeddingsA, ] + x_orig
        else:
            x = x_orig

        x = [torch.cat([x[i] * x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        x = [self.channel_att[i](x[i]) for i in range(len(x))]

        losses = dict()

        loss_decode = self._decode_head_forward_train_with_text(x, text_embeddingsA, text_embeddingsB, data_samples)
        losses.update(loss_decode)

        if self.with_identity_head:
            loss_identity_sm = self._identity_head_forward_train(
                score_map_diff / self.tau, data_samples, 'aux_score_map')
            losses.update(loss_identity_sm)
            loss_identity1 = self._identity_head_forward_train(
                x[0], data_samples, 'aux_layer0')
            losses.update(loss_identity1)
            # loss_identity1 = self._identity_head_forward_train(
            #     x[0], data_samples, 'aux_layer0')
            # losses.update(loss_identity1)
            loss_identity2 = self._identity_head_forward_train(
                x[1], data_samples, 'aux_layer1')
            losses.update(loss_identity2)
            loss_identity3 = self._identity_head_forward_train(
                x[2], data_samples, 'aux_layer2')
            losses.update(loss_identity3)
            loss_identity4 = self._identity_head_forward_train(
                x[3], data_samples, 'aux_layer3')
            losses.update(loss_identity4)
            # loss_identity4 = self._identity_head_forward_train(
            #     x[3], data_samples, 'aux_layer3')
            # losses.update(loss_identity4)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                _x_orig, data_samples)
            losses.update(loss_aux)

        return losses



    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
    
        inputsA = inputs[:, :3, :, :]  #inputsA :(1,3,256,256)
        inputsB = inputs[:, 3:, :, :]  #inputsB :(1,3,256,256)
        # xA = self.extract_feat(inputsA)  #resnet：1 64 64 64   1024  8 8
        # xB = self.extract_feat(inputsB)

        xA, outA ,memoryA,memory_textA,referenceA = self.extract_feat(inputsA)  #修改过
        xB,outB ,memoryB,memory_textB,referenceB = self.extract_feat(inputsB)

        memoryA=transform_features(memoryA)
        memoryB=transform_features(memoryB)

        # print(xA[0].size(), memoryA[0].size())

        # text_embeddingsA, x_dinoA,=self.M1(xA,memoryA)
        # # text_embeddingsB, x_dinoB,=self.M2(xB,memoryB)
        # text_embeddingsB, x_dinoB,=self.M1(xB,memoryB)




        text_embeddingsA, x_dinoA,=self.ATT1(xA,memoryA)
        # text_embeddingsB, x_dinoB,=self.M2(xB,memoryB)
        text_embeddingsB, x_dinoB,=self.ATT2(xB,memoryB)

        # text_embeddingsA, x_dinoA,=self.SemanticGuidanceModule(xA,memoryA)
        # text_embeddingsB, x_dinoB,=self.SemanticGuidanceModule(xB,memoryB)

        # for name, param in self.extract_feat.named_parameters():
        #     if param.requires_grad:
        #         print(f"Name: {name}, Shape: {param.shape}")

        #
        # possA[0]=self.FCovblock1(possA[0])
        # possA[1]=self.FCovblock2(possA[1])
        # possA[2]=self.FCovblock3(possA[2])
        # possA[3]=self.FCovblock4(possA[3])
        #
        # possB[0]=self.FCovblock1(possB[0])
        # possB[1]=self.FCovblock2(possB[1])
        # possB[2]=self.FCovblock3(possB[2])
        # possB[3]=self.FCovblock4(possB[3])



        x_g = xA[-1][0]+xB[-1][0]


        x_l = xA[-1][1]+xB[-1][1]




        # x_cat = [torch.cat((xA[i]+possA[i], xB[i]+possB[i]), dim=1) for i in range(len(xA)-1)]
        x_cat = [torch.cat((xA[i], xB[i]), dim=1) for i in range(len(xA)-1)]
        x_cat.append([x_g, x_l])









        # textA, textB = self.get_cls_text(data_samples) #1 2 64
        # text_embeddingsA, x_dinoA, score_mapA = self.after_extract_feat_dino(xA, textA) #1 2 1024   x_dinoA size： xA  1 2 8 8
        # text_embeddingsB, x_dinoB, score_mapB = self.after_extract_feat_dino(xB, textB)


        #
        # text_embeddingsA, x_dinoA, score_mapA = self.AfterExtractFeatDino(xA, memory_textA, referenceA[-1]) #1 2 1024   x_dinoA size： xA  1 2 8 8
        # text_embeddingsB, x_dinoB, score_mapB = self.AfterExtractFeatDino(xB, memory_textB, referenceB[-1])


        # text_embeddingsA, x_dinoA, = self.AfterExtractFeatDino(xA, memory_textA) #1 2 1024   x_dinoA size： xA  1 2 8 8
        # text_embeddingsB, x_dinoB, = self.AfterExtractFeatDino(xB, memory_textB)








        # text_embeddingsA, x_dinoA, score_mapA = self.after_extract_feat_dino(xA, memory_textA) #1 2 1024   x_dinoA size： xA  1 2 8 8
        # text_embeddingsB, x_dinoB, score_mapB = self.after_extract_feat_dino(xB, memory_textB)

        # visualization_tsne(x_dinoA[0],x_dinoB[0],data_samples)


        x_orig = [torch.cat([x_dinoA[i], x_dinoB[i]], dim=1) for i in range(len(x_dinoA))]  # 2* xA
        #
        # x_minus = [self.minus_conv[i](torch.abs(x_dinoA[i]-x_dinoB[i])) for i in range(len(x_dinoA))]
        # x_diff = [F.sigmoid(1-torch.cosine_similarity(x_dinoA[i], x_dinoB[i], dim=1)).unsqueeze(1) for i in range(len(x_dinoA))]
        x_minus = self.TF2(x_dinoA, x_dinoB, text_embeddingsA, text_embeddingsB,referenceA[-1],referenceB[-1])

        # visualization(inputsA, inputsB, x_minus[0], batch_img_metas)
        # visualization(inputsA, inputsB, x_minus[0], data_samples)

        # x_minus = self.TF1(x_dinoA, x_dinoB)
        x_diff = [torch.sigmoid(torch.mean((x_dinoA[i] - x_dinoB[i]) ** 2, dim=1, keepdim=True)) for i in
                  range(len(x_dinoA))]

        # score_map_diff = score_mapA-score_mapB

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()
        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        x = [self.channel_att[i](x[i]) for i in range(len(x))]

        losses = dict()

        loss_decode = self._decode_head_forward_train_with_text(x, text_embeddingsA, text_embeddingsB, referenceA[-1],referenceB[-1], data_samples)
        # loss_decode = self._decode_head_forward_train_with_text(x, text_embeddingsA, text_embeddingsB,data_samples)
        losses.update(loss_decode)

        # if self.with_identity_head:
        #     loss_identity_sm = self._identity_head_forward_train(
        #         score_map_diff/self.tau, data_samples, 'aux_score_map')
        #     losses.update(loss_identity_sm)
        #     loss_identity1 = self._identity_head_forward_train(
        #         x[0], data_samples, 'aux_layer0')
        #     losses.update(loss_identity1)
        #     # loss_identity1 = self._identity_head_forward_train(
        #     #     x[0], data_samples, 'aux_layer0')
        #     # losses.update(loss_identity1)
        #     loss_identity2 = self._identity_head_forward_train(
        #         x[1], data_samples, 'aux_layer1')
        #     losses.update(loss_identity2)
        #     loss_identity3 = self._identity_head_forward_train(
        #         x[2], data_samples, 'aux_layer2')
        #     losses.update(loss_identity3)
        #     loss_identity4 = self._identity_head_forward_train(
        #         x[3], data_samples, 'aux_layer3')
        #     losses.update(loss_identity4)
        #     # loss_identity4 = self._identity_head_forward_train(
        #     #     x[3], data_samples, 'aux_layer3')
        #     # losses.update(loss_identity4)
        #
        # if self.with_auxiliary_head:
        #     loss_aux = self._auxiliary_head_forward_train(
        #         _x_orig, data_samples)
        #     losses.update(loss_aux)

        return losses

    def loss_dino(self, inputs: Tensor, data_samples: SampleList) -> dict:

        losses = dict()

        # 1. 特征提取
        inputsA, inputsB = inputs[:, :3, :, :], inputs[:, 3:, :, :]
        xA, outA, memoryA, memory_textA, referenceA = self.extract_feat(inputsA)
        xB, outB, memoryB, memory_textB, referenceB = self.extract_feat(inputsB)

        change_pred, loss, loss_dict  = self.CD(
            xA, xB, outA, outB, memoryA, memoryB, memory_textA, memory_textB, referenceA, referenceB,data_samples)

        return loss

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        import time
        torch.cuda.synchronize()







        start = time.time()
        inputsA = inputs[:, :3, :, :]
        inputsB = inputs[:, 3:, :, :]
        xA = self.extract_feat(inputsA)
        xB = self.extract_feat(inputsB)

        x_g = xA[-1][0]+xB[-1][0]
        x_l = xA[-1][1]+xB[-1][1]
        x_cat = [torch.cat((xA[i], xB[i]), dim=1) for i in range(len(xA)-1)]
        x_cat.append([x_g, x_l])

        textA = []
        textB = []
        foreA = ', '.join(['remote sensing image foreground objects']+['mountain', 'bare land', 'ground track field', 'road', 'farmland', 'dense residential', 'island', 'highway', 'fertile land'])
        backA = ', '.join(['remote sensing image background objects'])
        foreB = ', '.join(['ground track field', 'farmland', 'bare land', 'wetland', 'golf course', 'island', 'fertile land', 'interchange', 'pond'])
        backB = ', '.join(['remote sensing image background objects'])
        textA.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backA, foreA]]).unsqueeze(0))
        textB.append(torch.cat([tokenize(c, context_length=self.context_length) for c in [backB, foreB]]).unsqueeze(0))
        textA, textB = torch.cat(textA, dim=0), torch.cat(textB, dim=0)

        text_embeddingsA, x_dinoA, score_mapA = self.after_extract_feat_dino(xA, textA)
        text_embeddingsB, x_dinoB, score_mapB = self.after_extract_feat_dino(xB, textB)

        x_orig = [torch.cat([x_dinoA[i], x_dinoB[i]], dim=1) for i in range(len(x_dinoA))]

        x_minus = [self.minus_conv[i](torch.abs(x_dinoA[i]-x_dinoB[i])) for i in range(len(x_dinoA))]
        x_diff = [F.sigmoid(1-torch.cosine_similarity(x_dinoA[i], x_dinoB[i], dim=1)).unsqueeze(1) for i in range(len(x_dinoA))]

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        if self.text_head:
            x = [text_embeddingsA,] + x_orig
        else:
            x = x_orig

        x = [torch.cat([x[i]*x_diff[i], x_minus[i], x[i]], dim=1) for i in range(len(x))]
        x = [self.channel_att[i](x[i]) for i in range(len(x))]
        data_samples = [{'image_shape': (256, 256)}]

        seg_logits = self.decode_head.predict_with_text(x, text_embeddingsA, text_embeddingsB, data_samples,
                                              self.test_cfg)
        torch.cuda.synchronize()
        end = time.time()
        total_time = end - start
        print('total_time:{:.2f}'.format(total_time))
        return seg_logits

    def mm_slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        inputs = inputs[0].unsqueeze(0)
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

        imgA_pil = Image.open(batch_img_metas[0]['img_path'])
        imgB_pil = Image.open(batch_img_metas[0]['img_path'].replace('/A', '/B'))

        model, preprocess = init_dino()

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                cropA = imgA_pil.crop((x1, y1, x2, y2))
                cropB = imgB_pil.crop((x1, y1, x2, y2))
                jsonA, jsonB = dino_infer(cropA, cropB, model, preprocess)
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                batch_img_metas[0]['jsonA'] = jsonA
                batch_img_metas[0]['jsonB'] = jsonB
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]


                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)

                from thop import profile

                # 假设 x 是一个示例输入
                # x = torch.randn(1, 3, 224, 224)  # 根据实际输入调整尺寸
                metas = ...  # 根据实际情况提供 batch_img_metas

                flops, params = profile(self.encode_decode, inputs=(crop_img, batch_img_metas))
                print(f"FLOPs: {flops}")
                print(f"Parameters: {params}")




                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole', 'mm_slide'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        elif self.test_cfg.mode == 'mm_slide':
            seg_logit = self.mm_slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
