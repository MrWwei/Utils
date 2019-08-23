#   _*_ coding:utf-8 _*_
__author__ = 'admin'

from PIL import Image, ImageDraw, ImageFont

img = Image.open('test1.png')
#   设置抠图区域
# box = [img.size[0] / 4, img.size[1] / 4, img.size[0] * 3 / 4, img.size[1] * 3 / 4]
box = [800,300,1000,800]
#   从图片上抠下此区域

region = img.crop(box)
# crop = region.load()
width = region.size[0]  # 获取宽度
height = region.size[1]  # 获取长度
for x in range(width):
    for y in range(height):
        region.putpixel((x, y), (11, 11, 11))
#   将此区域旋转180度
# region = region.transpose(Image.ROTATE_180)
#   查看抠出来的区域
# region.show()
#   将此区域粘回去
img.paste(region, box)
img.save('region.png')
# img.show()

