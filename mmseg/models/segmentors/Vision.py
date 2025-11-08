import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from sklearn.manifold import TSNE
import cv2
from dataclasses import dataclass
import time
import seaborn as sns
# 创建一个简化的SegDataSample类
@dataclass
class SegDataSample:
    img_path: str
    gt_sem_seg: torch.Tensor
    img_shape: tuple
    flip: bool = False




import time

# 获取当前时间并格式化
current_time = time.strftime('%Y%m%d%H%M%S')
print(time.strftime('%Y%m%d%H%M%S'))  # 会打印类似：20241023171923





def create_mock_data(batch_size=16):
    """创建模拟数据"""
    # 创建随机图像数据
    pre_images = torch.randn(batch_size, 3, 256, 256)
    post_images = torch.randn(batch_size, 3, 256, 256)

    # 创建模拟的data samples
    data_samples = []
    for i in range(batch_size):
        # 创建随机的二值标签
        label = torch.randint(0, 2, (1, 256, 256))
        # 创建模拟的文件路径
        img_path = f"/mock/data/path/image_{i:03d}.png"

        data_samples.append(SegDataSample(
            img_path=img_path,
            gt_sem_seg=label,
            img_shape=(256, 256)
        ))



    Min = torch.randn(16, 256, 64, 64)

    return pre_images, post_images, Min,data_samples


class FeatureVisualizer:
    def __init__(self, save_dir=str("heatmap/"+time.strftime('%Y%m%d%H%M%S'))+'visualization_results'):
        # 创建基本的特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 创建保存目录
        self.save_dir = Path(save_dir)
        self.heatmap_dir =  self.save_dir / 'heatmaps'
        self.tsne_dir = self.save_dir / 'tsne'
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
        self.tsne_dir.mkdir(parents=True, exist_ok=True)

    def extract_features(self, x):
        """提取特征"""
        return self.feature_extractor(x)

    def generate_difference_heatmap(self,pre_images,post_images, Min, img_name):
        """生成并保存差异热力图"""
        # 计算特征差异
        diff = Min
        diff = torch.mean(diff, dim=1).squeeze().detach().cpu().numpy()

        # 归一化
        diff = (diff - diff.min()) / (diff.max() - diff.min())

        # 调整大小到原始图像尺寸
        diff_resized = cv2.resize(diff, (512, 512))

        # 使用jet colormap
        heatmap = cv2.applyColorMap((diff_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        timestamp = time.strftime('%Y%m%d%H%M%S')
        save_path = self.heatmap_dir / f"{timestamp}_{img_name}_diff_heatmap.png"
        # 保存热力图
        # save_path = self.heatmap_dir/time.strftime('%Y%m%d%H%M%S')+f"{img_name}_diff_heatmap.png"
        # save_path_pre = self.heatmap_dir / f"{img_name}_save_path_pre.png"
        # save_path_post = self.heatmap_dir / f"{img_name}_save_path_post.png"
        cv2.imwrite(str(save_path), heatmap)
        # cv2.imwrite(str(save_path_pre),  (pre_images*255).squeeze(0).permute(1, 2, 0).cpu().numpy())
        # cv2.imwrite(str(save_path_post),  (post_images*255).squeeze(0).permute(1, 2, 0).cpu().numpy())

        return diff_resized

    def generate_tsne(self, Min, img_name):
        """生成并保存t-SNE可视化"""
        # 处理特征
        # pre_feat = pre_feat.squeeze().permute(1, 2, 0).reshape(-1, pre_feat.size(1))
        # post_feat = post_feat.squeeze().permute(1, 2, 0).reshape(-1, post_feat.size(1))

        Min=Min.reshape(256, -1)


        # 采样点（减少计算量）
        # n_samples = 20000
        # if len(pre_feat) > n_samples:
        #     idx1 = np.random.choice(len(pre_feat), n_samples, replace=False)
        #     idx2 = np.random.choice(len(post_feat), n_samples, replace=False)
        #     pre_feat = pre_feat[idx1]
        #     post_feat = post_feat[idx2]
        #
        # # 合并特征并转换为numpy
        # features = torch.cat([pre_feat, post_feat], dim=0).detach().cpu().numpy()
        # labels = np.array([0] * len(pre_feat) + [1] * len(post_feat))

        # t-SNE降维
        tsne = TSNE(n_components=2,  # 增加perplexity
        n_iter=1000,    # 增加迭代次数
        learning_rate='auto')
        embeddings = tsne.fit_transform(Min)

        # 创建图像
        plt.figure(figsize=(10, 10))
        plt.scatter(embeddings[label==0, 0], embeddings[:len(pre_feat), 1],
                    c='blue', alpha=0.3, s=2,rasterized=True)
        plt.scatter(embeddings[len(pre_feat):, 0], embeddings[len(pre_feat):, 1],
                    c='red', alpha=0.3, s=2,rasterized=True)
        plt.axis('off')
        plt.legend()
        plt.gca().get_legend().remove()
        # 保存图像
        save_path = self.tsne_dir / f"{img_name}_tsne.png"
        plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close()


def visualization(pre_images, post_images,Min,data_samples):
    # 生成模拟数据


    # 创建可视化器
    visualizer = FeatureVisualizer()



    # 遍历每对图像
    for i in range(len(pre_images)):
        # 获取图像名称
        # img_name = Path(data_samples[i].img_path).stem
        # print(data_samples[i])


        img_name = os.path.splitext(os.path.basename(data_samples[i]['img_path']))[0]


        # print(f"Processing {img_name}...")
        print(f"Processing {img_name}...")

        # 提取特征
        # with torch.no_grad():
        #     pre_feat = visualizer.extract_features(pre_images[i:i + 1])
        #     post_feat = visualizer.extract_features(post_images[i:i + 1])

        # 生成差异热力图


        visualizer.generate_difference_heatmap(pre_images[i:i + 1],post_images[i:i + 1],Min[i:i + 1], img_name)

        # 生成t-SNE图
        # visualizer.generate_tsne(Min, img_name)

        print(f"Saved visualizations for {img_name}")


if __name__ == "__main__":
    pre_images, post_images,Min, data_samples = create_mock_data(batch_size=16)
    visualization(pre_images, post_images, Min, data_samples)