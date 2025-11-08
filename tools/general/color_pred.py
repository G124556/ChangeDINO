import os
import cv2
import numpy as np

p = '/home/user/ChangeDINO/work_dirs/changedino_sysu/test_result_src'
save_result = '/home/user/ChangeDINO/work_dirs/changedino_sysu/test_result_src_vis'
os.makedirs(save_result, exist_ok=True)
files = os.listdir(p)
for item in files:
    img = cv2.imread(os.path.join(p, item), 0)
    img_vis = np.where(img > 0, 255, 0)
    cv2.imwrite(os.path.join(save_result, item), img_vis)
