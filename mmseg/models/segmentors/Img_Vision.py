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

    return pre_images, post_images, data_samples


class FeatureVisualizer:
    def __init__(self, save_dir=str("TSNE500/") + str(time.strftime('%Y%m%d%H%M%S'))+'visualization_results'):
        # 创建基本的特征提取网络
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )

        # 创建保存目录
        self.save_dir = Path(save_dir)
        self.heatmap_dir =  self.save_dir / 'heatmaps'
        self.tsne_dir = self.save_dir / 'tsne'
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
        self.tsne_dir.mkdir(parents=True, exist_ok=True)

    def extract_features(self, x):
        """提取特征"""
        # return self.feature_extractor(x)

    def generate_difference_heatmap(self,pre_images,post_images,pre_feat, post_feat, img_name):
        """生成并保存差异热力图"""
        # 计算特征差异
        diff = torch.abs(pre_feat - post_feat)
        diff = torch.mean(diff, dim=1).squeeze().detach().cpu().numpy()

        # 归一化
        diff = (diff - diff.min()) / (diff.max() - diff.min())

        # 调整大小到原始图像尺寸
        diff_resized = cv2.resize(diff, (512, 512))

        # 使用jet colormap
        heatmap = cv2.applyColorMap((diff_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # 保存热力图
        save_path = self.heatmap_dir / f"{img_name}_diff_heatmap.png"
        save_path_pre = self.heatmap_dir / f"{img_name}_save_path_pre.png"
        save_path_post = self.heatmap_dir / f"{img_name}_save_path_post.png"
        cv2.imwrite(str(save_path), heatmap)
        cv2.imwrite(str(save_path_pre),  pre_images.squeeze(0).permute(1, 2, 0).cpu().numpy())
        cv2.imwrite(str(save_path_post),  post_images.squeeze(0).permute(1, 2, 0).cpu().numpy())

        return diff_resized

    def generate_tsne(self, pre_feat, post_feat, img_name, Data):
        """
        生成空白区域更少的t-SNE可视化图
        """
        # 处理特征维度
        pre_feat = pre_feat.squeeze().permute(1, 2, 0).reshape(-1, pre_feat.size(1))
        post_feat = post_feat.squeeze().permute(1, 2, 0).reshape(-1, post_feat.size(1))

        # 固定随机种子


        # 采样点
        n_samples = 10000
        if len(pre_feat) > n_samples:
            idx1 = np.random.choice(len(pre_feat), n_samples, replace=False)
            idx2 = np.random.choice(len(post_feat), n_samples, replace=False)
            pre_feat = pre_feat[idx1]
            post_feat = post_feat[idx2]

        # 合并特征并转换为numpy
        features = torch.cat([pre_feat, post_feat], dim=0).detach().cpu().numpy()
        labels = np.array([0] * len(pre_feat) + [1] * len(post_feat))

        # 标准化特征
        features = (features - features.mean(axis=0)) / features.std(axis=0)

        # 配置t-SNE，减小perplexity使聚类更紧凑
        tsne = TSNE(
            n_components=2,
            perplexity=25,  # 稍微减小perplexity
            n_iter=500,
            learning_rate='auto',
            init='pca',
        )

        embeddings = tsne.fit_transform(features)

        # 计算显示范围，减小边界留白
        buffer = 0.02  # 减小边界留白
        x_min, x_max = embeddings[:, 0].min(), embeddings[:, 0].max()
        y_min, y_max = embeddings[:, 1].min(), embeddings[:, 1].max()

        # 确保正方形显示区域
        total_range = max(x_max - x_min, y_max - y_min)
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        x_min = x_center - total_range / 2 - total_range * buffer
        x_max = x_center + total_range / 2 + total_range * buffer
        y_min = y_center - total_range / 2 - total_range * buffer
        y_max = y_center + total_range / 2 + total_range * buffer

        # 创建图像，减小边距
        plt.style.use('default')
        fig = plt.figure(figsize=(10, 10))
        # 增大绘图区域，减少边距
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])

        # 设置图表样式
        ax.set_aspect('equal')

        # 绘制散点图，增大点的大小
        ax.scatter(embeddings[:len(pre_feat), 0], embeddings[:len(pre_feat), 1],
                   c='blue', alpha=0.4, s=4, rasterized=True)
        ax.scatter(embeddings[len(pre_feat):, 0], embeddings[len(pre_feat):, 1],
                   c='red', alpha=0.4, s=4, rasterized=True)

        # 设置固定的显示范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # 移除坐标轴和刻度
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        # 设置边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)

        # 保存图像，减小内边距
        temp_path = self.tsne_dir / f"{img_name}_temp_tsne.png"
        plt.savefig(temp_path,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.1,  # 减小内边距
                    facecolor='white',
                    edgecolor='none')
        plt.close()

        # 使用PIL精确调整图像大小
        from PIL import Image
        img = Image.open(temp_path)
        img = img.resize((3000, 3000), Image.Resampling.LANCZOS)

        # 保存最终图像
        final_path = self.tsne_dir / f"{img_name}_tsne.png"
        img.save(final_path, dpi=(300, 300))

        # 删除临时文件
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)


def visualization_tsne(pre_images, post_images, data_samples):
    # 生成模拟数据
    # pre_images, post_images, data_samples = create_mock_data(batch_size=16)

    # 创建可视化器
    visualizer = FeatureVisualizer()

    # 遍历每对图像
    for i in range(len(pre_images)):
        # 获取图像名称
        # img_name = Path(data_samples[i].img_path).stem

        img_name = os.path.splitext(os.path.basename(data_samples[i]['img_path']))[0]  # 得到 'test_1'


        print(f"Processing {img_name}...")

        # 提取特征
        # with torch.no_grad():
        #     pre_feat = visualizer.extract_features(pre_images[i:i + 1])
        #     post_feat = visualizer.extract_features(post_images[i:i + 1])

        # 生成差异热力图


        # visualizer.generate_difference_heatmap(pre_images[i:i + 1],post_images[i:i + 1],pre_feat, post_feat, img_name)

        # 生成t-SNE图
        visualizer.generate_tsne(pre_images[i], post_images[i], img_name,data_samples[i])

        print(f"Saved visualizations for {img_name}")


if __name__ == "__main__":
    visualization_tsne()