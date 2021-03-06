# /home/em/disk/wxyt40/radar_zsss_cap_20170112/img/SHA170112000052.png
# 40*720*720*3
import cv2
import numpy as np
import os
from tqdm import tqdm
import multiprocessing


def combine(img_path, target, img_name):
    img = cv2.imread(img_path, 0)
    img_list = []
    for i in range(40):
        img_s = img[:, i * 720:(i + 1) * 720]
        img_list.append(img_s)
    max_img1 = np.array(img_list)

    # img_e = np.zeros((720, 720, 1))
    img_e = max_img1.max(axis=0)
    img_e[img_e < 84] = 0
    img_e[img_e == 255] = 0

    if not os.path.exists(target):
        os.makedirs(target)
    target_path = os.path.join(target, img_name)
    cv2.imwrite(target_path, img_e)


root = '/data2/wxyt/CAPPI_250'
days = os.listdir(root)
target_root = '/data2/wxyt/CAPPI_250_sl'


def process(queue):
    while(True):
	
    	day = queue.get()
    	target = os.path.join(target_root, day)
    	imgs_path = os.path.join(root, day, 'img')
    	imgs = [i for i in os.listdir(imgs_path) if i.endswith(".png")]

   	for img in imgs:
            img_path = os.path.join(imgs_path, img)
            combine(img_path, target, img)


p_num = 32
queue = multiprocessing.Queue(32)

p_list = [multiprocessing.Process(target=process, args=(queue,)) for i in range(p_num)]

for p in p_list:
    p.start()
for day in tqdm(days):
    queue.put(day)

for i in range(p_num):
    queue.put(None)
