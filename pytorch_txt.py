#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

sameples = '/home/em/disk/work/pytorch/data/wcedata/'
traintxt = open('/home/em/disk/work/pytorch/data/wcedata/train.txt', 'w')
valtxt = open('/home/em/disk/work/pytorch/data/wcedata/val.txt', 'w')

for root, dirs, files in os.walk(sameples):
    for file in files:
        label = file.split('.')[0]
        if label == 'bleeding':
            label = 0
        elif label == 'colon':
            label = 1
        else:
            label = 2
        if root.split('/')[-2] == 'train':
            traintxt.write(
                './data/wcedata/train/' + root.split('/')[-1] + '/' + file + '\t' + str(label) + '\n')
        if root.split('/')[-2] == 'val':
            valtxt.write('./data/wcedata/val/' + root.split('/')[-1] + '/' + file + '\t' + str(label) + '\n')
