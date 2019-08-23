#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import cv2
from tqdm import tqdm

'''
ppi数据切割、分组
'''


def touch_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def mv_cut(root_path, target_path):
    days = [i for i in os.listdir(root_path) if i.startswith("radar")]

    for day in tqdm(days):
        root_day = os.path.join(root_path, day, "img")

        # target_days = ['radar_zsss_raw_20170112_L', 'radar_zsss_raw_20170112_M', 'radar_zsss_raw_20170112_S']
        target_days = ['L' + '/' + day + '_L', 'M' + '/' + day + '_M', 'S' + '/' + day + '_S']
        target_dirs = []

        for name in target_days:
            tar_day = touch_dir(os.path.join(target_path, name))
            target_dirs.append(tar_day)

        imgs = [i for i in os.listdir(root_day) if i.endswith(".png")]
        for imgname in imgs:
            # img = cv2.imread(os.path.join(root_day, imgname))[:, :1662, :]
            img = cv2.imread(os.path.join(root_day, imgname))
            img_1 = img[:, :1661, :]
            img_2 = img[:, 1662:3323, :]
            img_3 = img[:, 3324:4985, :]
            path1 = target_dirs[0] + '/' + imgname
            path2 = target_dirs[1] + '/' + imgname
            path3 = target_dirs[0] + '/' + imgname
            cv2.imwrite(path1, img_1)
            cv2.imwrite(path2, img_2)
            cv2.imwrite(path3, img_3)


root = '/data1/ppi'
target = '/data1/ppi_new'
mv_cut(root, target)
