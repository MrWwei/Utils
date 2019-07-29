#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# img = np.ones((1024, 2048, 3), dtype=np.uint8)  # random.random()方法后面不能加数据类型
# img = np.random.random((3,3)) #生成随机数都是小数无法转化颜色,无法调用cv2.cvtColor函数
# img[:, :, 0] = 142
# img[:, :, 1] = 0
# img[:, :, 2] = 0
# img = cv2.rectangle(img, (384, 100), (510, 200), (0, 0, 120), -1)
# img = cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)

# img = cv2.imread('/data1/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png')
# imgs = os.listdir('/data1/cityscapes/gtFine/val/frankfurt')
# img = cv2.imread('/home/em/Pictures/test11.png')
# for img in imgs:
#     img_path = os.path.join('/data1/cityscapes/gtFine/val/frankfurt', img)
#     img = cv2.imread(img_path)
#     img[:, :, 1] = img[:, :, 0]
#     img[:, :, 2] = img[:, :, 0]
#     cv2.imwrite(img_path, img)

# plt.imshow(img)
# plt.show()
# shape = img.shape
# cv2.imshow('img', img)
# cv2.imwrite('img.png', img)

# cv2.waitKey(0)
img = cv2.imread('/home/em/Pictures/test12.png')


# test = img[img == [128, 128, 128]]
# img[img == [128, 128, 128]]
# img[img == [128, 0, 0]] = 7
# test2 = img[img == [0, 0, 128]]
cv2.imwrite('/home/em/Pictures/input1.png', img)
