import numpy as np
import matplotlib.pyplot as plt


def calculate_and_visualize_mi(matrix1, matrix2, bins=9):
    """计算互信息并可视化联合分布"""
    x = matrix1.flatten()
    y = matrix2.flatten()

    # 计算联合分布
    joint_hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

    # 归一化得到概率
    joint_prob = joint_hist / np.sum(joint_hist)
    x_prob = np.sum(joint_prob, axis=1)  # 边缘概率x
    y_prob = np.sum(joint_prob, axis=0)  # 边缘概率y

    # 计算互信息
    mi = 0
    contribution_matrix = np.zeros_like(joint_prob)  # 用于可视化每个点的贡献

    for i in range(bins):
        for j in range(bins):
            if joint_prob[i, j] > 0:
                # 计算每个点对互信息的贡献
                contribution = joint_prob[i, j] * np.log2(
                    joint_prob[i, j] / (x_prob[i] * y_prob[j])
                )
                mi += contribution
                contribution_matrix[i, j] = contribution

    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 联合分布图
    im1 = ax1.imshow(joint_prob, cmap='Blues')
    ax1.set_title('联合分布')
    plt.colorbar(im1, ax=ax1)

    # 互信息贡献图
    im2 = ax2.imshow(contribution_matrix, cmap='RdBu')
    ax2.set_title('每个点对互信息的贡献')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.show()

    return mi


# 测试案例1：完全相关的矩阵
matrix1 = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
matrix2 = matrix1.copy()  # 完全相同的矩阵

print("案例1：完全相关的矩阵")
mi1 = calculate_and_visualize_mi(matrix1, matrix2)
print(f"互信息值：{mi1:.4f}")

# 测试案例2：部分相关的矩阵
matrix3 = matrix1 + np.random.normal(0, 0.5, matrix1.shape)

print("\n案例2：部分相关的矩阵")
mi2 = calculate_and_visualize_mi(matrix1, matrix3)
print(f"互信息值：{mi2:.4f}")

# 测试案例3：完全不相关的矩阵
matrix4 = np.random.rand(3, 3)

print("\n案例3：完全不相关的矩阵")
mi3 = calculate_and_visualize_mi(matrix1, matrix4)
print(f"互信息值：{mi3:.4f}")