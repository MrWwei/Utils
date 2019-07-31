# !python3
# coding:utf8
import numpy as np
from PIL import Image
import cv2


# 两种修改像素方法：1.=赋值 2.putpixel

def map(imgpath):
    # im = np.array(Image.open('test3.png'))  # 打开图片
    im = Image.open('color.png')
    im1 = cv2.imread('test3.png')
    # im = Image.new("RGB", (400, 400), (255, 0, 0))  # 红色

    width = im.size[0]  # 获取宽度

    height = im.size[1]  # 获取长度
    pim = im.load()

    for i in range(width):
        for j in range(height):
            if pim[i, j] == (192, 192, 0):  # car
                pim[i, j] = (26, 26, 26)
            if pim[i, j] == (64, 0, 0):  # road
                pim[i, j] = (7, 7, 7)
            if pim[i, j] == (0, 128, 0):
                pim[i, j] = (1, 1, 1)

        # test = pim[i, j]
        # if pim[i, j] == 3:  # 绿色
        #     pim[i, j] = 192

    im.save("a.png")


if __name__ == '__main__':
    im = Image.open('color.png')
    im.show()

    pix = im.load()  # 导入像素
    width = im.size[0]  # 获取宽度
    height = im.size[1]  # 获取长度

    for x in range(width):
        for y in range(height):
            r, g, b = im.getpixel((x, y))
            rgba = (r, g, b)
            if (r == 64 and g == 0 and b == 128):
                im.putpixel((x, y), (11, 11, 11))
            # if (a == 255):
            #     im.putpixel((x, y), (255, 255, 255, 255))

    im = im.convert('RGB')
    im.save('456.png')
