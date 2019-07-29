#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import cv2
import argparse
import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='s1、s2、s3， white to black')
parser.add_argument('--input', type=str, default='/data1/data2/wxyt/split_sl/s2', help='')
args = parser.parse_args()
for root, dirs, files in os.walk(args.input):
    for file in tqdm.tqdm(files):
        img_path = os.path.join(root, file)
        im = cv2.imread(img_path)
        # test = np.max(im)
        # try:
        im[im == 255] = 0
        # except:
        #     print('Error:' + img_path)
        #     os.rmdir(root)
        #     print('dir has been removed!')
        #     pass
        # continue
        cv2.imwrite(img_path, im)
