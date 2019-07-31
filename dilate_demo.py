'''
    数学形态学 膨胀 dilate
'''
import cv2
import numpy as np

# 读入灰度图
img = cv2.imread("test1.png", flags=cv2.IMREAD_GRAYSCALE)

# 创建 核
kernel = np.ones((3,3), np.uint8)
# 膨胀
dilate_img = cv2.dilate(img, kernel, iterations=8)

cv2.imwrite('dao_dilate_k5.png', dilate_img)
# cv2.imwrite('dao_dilate_k5.png', np.hstack((img, dilate_img)))