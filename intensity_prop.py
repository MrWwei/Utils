#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
强度分割，强度大于30
'''
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import argparse


def main(args):
    # i 每隔20张取40张
    # root = '/data2/wxyt/CAPPI_250_combine'
    root = args.root
    # root = '/data1/ppi'
    # target = '/data1/ppi_sl'
    # target = '/data2/wxyt/CAPPI_250_sl_40'
    target = args.target
    days = os.listdir(root)

    props = []
    threshold = args.threshold
    for day in tqdm(days):
        lst = sorted(os.listdir(os.path.join(root, day)))
        count = 0
        for i in range(1, len(lst) - 40, 20):
            count += 1
            target_dir = os.path.join(target, day + '_' + str(count))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                # 40张图片，先判断平均占比，如果满足阈值则保留
            for j in range(i, i + 40):
                file = root + '/' + day + '/' + lst[j]
                # 按照阈值
                # 计算占比
                img = cv2.imread(file)
                all = (img.reshape(-1, 3)[:, 1] > -100).sum()
                max = np.max(img[:, :, 0])
                loc = (img.reshape(-1, 3)[:, 1] >= threshold).sum()
                proportion = loc / all
                props.append(proportion)
                img[img < threshold] = 0
                # print('target:',os.path.join(target_dir, file))
                cv2.imwrite(os.path.join(target_dir, lst[j]), img)
                # shutil.copy(file, target_dir)
            # if average<0.5
            average = sum(props) / 40

            print(target_dir + ':', str(average))
            props.clear()
            if average < 0.05:
                shutil.rmtree(target_dir)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='split intensity')
    parser.add_argument("--root", type=str, default='/data2/wxyt/CAPPI_250_combine')
    parser.add_argument("--target", type=str, default='/data2/wxyt/CAPPI_250_combine_target')
    parser.add_argument("--threshold", type=int, default=124)
    parser.add_argument("--percent", type=float, default=0.01)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
