#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
雷达拼图分割到文件夹
'''
import os
import shutil

# i 每隔20张取40张
root = '/data2/wxyt/CAPPI_250_sl'
# root = '/data1/ppi'
# target = '/data1/ppi_sl'
target = '/data2/wxyt/CAPPI_250_sl_40'
days = os.listdir(root)
for day in days:
    lst = sorted(os.listdir(os.path.join(root, day)))
    count = 0
    for i in range(1, len(lst) - 40, 20):
        count += 1
        target_dir = os.path.join(target, day + '_' + str(count))
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for j in range(i, i + 40):
            file = root + '/' + day + '/' + lst[j]
            shutil.copy(file, target_dir)
