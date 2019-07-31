#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from tqdm import tqdm
import heapq
import cv2
import os
# os.()

def cv_readlist(path_list, channel=-1):
    img_list = []
    # for path in path_list:
    for path in tqdm(path_list):
        # if i > 10: break
        if channel == -1:
            img_list.append(cv2.imread(path)[:, :, 0])
        else:
            img_list.append(cv2.imread(path)[:, channel * 720:channel * 720 + 720, 0])
    return img_list


def touch_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def getListMaxNumIndex(num_list,topk=5):
    max_num_index=list(map(num_list.index, heapq.nlargest(topk,num_list)))
    return max_num_index


def avg_list(lst):
    if len(lst) == 0: return None
    return sum(lst) / float(len(lst))