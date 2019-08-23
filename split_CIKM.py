#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import shutil
from tqdm import tqdm

root = '/data1/CIKM/data_new/CIKM2017_train/img2'
lst = sorted(os.listdir(root))

for x in tqdm(range(1, 10000)):
    for y in range(1, 5):
        target = '/data1/split_CIKM/' + str(x) + '_' + str(y)
        if not os.path.exists(target):
            os.mkdir(target)
        for z in range(1, 16):
            try:
                shutil.copy(os.path.join(root, str(x) + '_' + str(z) + '_' + str(y) + '.png'), target)
            except:
                if os.path.exists(target):
                    os.rmdir(target)
                continue



