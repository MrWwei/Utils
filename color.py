# /data1/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png
import numpy as np
import cv2


def checkGray(chip):
    chip_gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
    r, g, b = cv2.split(chip)
    r = r.astype(np.float32)
    g = g.astype(np.float32)
    b = b.astype(np.float32)
    s_w, s_h = r.shape[:2]
    x = (r + b + g) / 3
    # x = chip_gray
    r_gray = abs(r - x)
    g_gray = abs(g - x)
    b_gray = abs(b - x)
    r_sum = np.sum(r_gray) / (s_w * s_h)
    g_sum = np.sum(g_gray) / (s_w * s_h)
    b_sum = np.sum(b_gray) / (s_w * s_h)
    gray_degree = (r_sum + g_sum + b_sum) / 3
    if gray_degree < 10:
        print("Gray")
    else:
        print("NOT Gray")


def checkGray1(chip):
    img_hsv = cv2.cvtColor(chip, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    s_w, s_h = s.shape[:2]
    s_sum = np.sum(s) / (s_w * s_h)
    if s_sum > 10:
        print("Not Gray")
        pass
    else:
        print('Gray')


img = cv2.imread('/data1/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png')
checkGray1(img)
