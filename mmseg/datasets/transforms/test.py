
import numpy as np

from osgeo import gdal

ds2 = gdal.Open("/data2/gaoyupeng/LESPS-master/ALL_data/whu-cd/val/post/1.png")




img1 = np.einsum('ijk->jki', ds2.ReadAsArray())


print(img1)

import numpy as np
from PIL import Image

# 读取图像
img = Image.open("/data2/gaoyupeng/LESPS-master/ALL_data/whu-cd/val/post/1.png")

# 将图像转换为NumPy数组并调整维度顺序
img_array = np.array(img)
img_array = np.transpose(img_array, (1, 2,0))

print(img_array.shape)  # 应该输出 (512, 512, 3)
print(img_array)