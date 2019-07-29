# -*- coding: utf-8 -*-
import os
import random
import argparse

parser = argparse.ArgumentParser(description='create dataset txt')
parser.add_argument('--root', type=str, default='/data1/ws_fast/dataset/Y027/wxyt/split_sl/s1')
parser.add_argument('--output', type=str, default='/home/em')
args = parser.parse_args()

dirs = os.listdir(args.root)
total_num = len(dirs)
train_num = int(0.8 * total_num)
print('train_num:%d' % train_num + '\n')
val_num = int(0.05 * total_num)
print('val_num:%d' % val_num + '\n')
test_num = int(0.1 * total_num)
print('test_num: %d' % test_num + '\n')
vis_num = int(0.05 * total_num)
print('vis_num: %d' % val_num + '\n')

# random.choice(train_num)
# train dataset
train_samples = random.sample(dirs, train_num)
# 删除列表中对应的元素
train_txt = open(os.path.join(args.output, 'train_data_list.txt'), 'w')
for train in train_samples:
    train_txt.write(str(train) + ' ' + str(40) + '\n')
    dirs.remove(train)
# val dataset
val_samples = random.sample(dirs, val_num)
val_txt = open(os.path.join(args.output, 'val_data_list.txt'), 'w')
for val in val_samples:
    val_txt.write(str(val) + ' ' + str(40) + '\n')
    dirs.remove(val)

# test dataset
test_samples = random.sample(dirs, test_num)
test_txt = open(os.path.join(args.output, 'test_data_list.txt'), 'w')
for test in test_samples:
    test_txt.write(str(test) + ' ' + str(40) + '\n')
    dirs.remove(test)

# vis dataset
vis_txt = open(os.path.join(args.output, 'vis_data_list.txt'), 'w')
for vis in dirs:
    vis_txt.write(str(vis) + ' ' + str(40) + '\n')
    dirs.remove(vis)
