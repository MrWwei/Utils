#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import matplotlib.pyplot as plt
from queue import Queue
import numpy as np
import argparse
import cv2
import os

import lib


def read_diffpic(new_paths, old_paths, old_imgs, pic_layer):
    if old_imgs == None:
        return lib.cv_readlist(new_paths, pic_layer), new_paths
    diff_paths, same_imgs = [], []
    for new_path in new_paths:
        if new_path in old_paths:
            same_imgs.append(old_imgs[old_paths.index(new_path)])
        else:
            diff_paths.append(new_path)
    diff_imgs = lib.cv_readlist(diff_paths, pic_layer)
    return same_imgs + diff_imgs, new_paths


def group_list(args):
    path = args.root_path
    group_gap = args.group_gap
    group_len = args.group_len
    pic_layer = args.pic_layer

    months = [int(i) for i in args.months.strip().split(",")]
    years = [int(i) for i in args.year.strip().split(",")]
    print(months)
    day_list = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]
    day_list = sorted(day_list)
    tmp_queue, count_start = Queue(), 0
    paths, img = None, None
    for day in day_list:
        count = 0
        if int(day[-4:-2]) not in months: continue
        if int(day[-6:-4]) not in years: continue
        img_path = os.path.join(path, day, "img")
        path_list = [i for i in os.listdir(img_path) if i.endswith(".png")]
        path_list = [os.path.join(img_path, i) for i in sorted(path_list)]

        tmp_Ilist, cur_Ilist, tmp_Plist, cur_paths = None, None, None, None
        while tmp_queue.qsize() != 0:
            tmp_Plist_ = tmp_queue.get()
            cur_paths_ = path_list[0:group_len - len(tmp_Plist_)]

            tmp_Ilist, tmp_Plist = read_diffpic(tmp_Plist_, tmp_Plist, tmp_Ilist, pic_layer)
            cur_Ilist, cur_paths = read_diffpic(cur_paths_, cur_paths, cur_Ilist, pic_layer)

            img = tmp_Ilist + cur_Ilist
            paths = tmp_Plist + cur_paths
            # ystr = day+"_"+str(count)
            # if len(paths) != group_len:
            #     ystr = day+"_"+str(count)+"_"+str(len(path))
            yield (paths, img, day)
            count += 1

        paths, img = None, None
        for start in range(0, len(path_list), group_gap):
            paths_ = path_list[start:start + group_len]
            img, paths = read_diffpic(paths_, paths, img, pic_layer)

            if len(paths) == group_len:
                yield (paths, img, day + "_" + str(count))
                count += 1
            elif len(paths) < group_len:
                tmp_queue.put((paths))


def read_pic(args):
    percent = args.percent
    target_path = lib.touch_dir(args.target_path)
    groups = group_list(args)
    sum_num = 720 * 720 * args.group_len

    # list_file = open("group_percent.lst", "w")
    count = 0
    for group in groups:
        paths, img_list, day = group
        img_array = np.array(img_list)

        # --------------------------------
        current_percent = np.count_nonzero(img_array) / sum_num
        # list_file.write(paths[0][-16:-4] + " " + str(round(current_percent, 2)) + "\n")
        # list_file.write(day + " " + str(round(current_percent, 2)) + "\n")
        # count+=1
        # if count % 500 ==0:
        #     print("write: ", day)
        # continue  # ----------------------

        if current_percent < percent:
            target_path_ = lib.touch_dir(os.path.join(target_path, "low"))
        else:
            target_path_ = lib.touch_dir(os.path.join(target_path, "high"))
        group_path = lib.touch_dir(os.path.join(target_path_, day + "_" + str(int(current_percent * 100))))

        for old_path, img_array in zip(paths, img_list):
            img_path = os.path.join(group_path, os.path.basename(old_path))
            cv2.imwrite(img_path, img_array)
        print("write: ", day)
        count += 1
    # list_file.close()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='split radar')
    parser.add_argument("--root_path", type=str, default="/data2/wxyt/CAPPI_250/")
    parser.add_argument("--target_path", type=str, default="/data2/wxyt/split_radar/rain")
    parser.add_argument("--months", type=str, default="4,5,6,7,8,9", help="5,6,7,8,9")
    parser.add_argument("--year", type=str, default="17,18,19", help="")
    parser.add_argument("--percent", type=float, default=0.24)
    parser.add_argument("--group_gap", type=int, default=20)
    parser.add_argument("--group_len", type=int, default=40)
    parser.add_argument("--pic_layer", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    read_pic(args)
