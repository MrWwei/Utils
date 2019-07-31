import numpy as np
import cv2
import os
from tqdm import tqdm
import bz2
import argparse

'''
解析北京和广州雷达回波数据
'''


def parseBin(input, output):
    files = os.listdir(input)
    count = 0
    for file in tqdm(files):
        f = open(os.path.join(input, file), 'rb')
        # 查看源数据是否有小数
        for i in f.read():
            if not isinstance(i, int):
                print(i)

        img = np.zeros((400, 400, 1)) if args.name == 'bj' else np.zeros((400, 430, 1))
        # img = np.zeros((400, 400, 1))  # 北京
        # img = np.zeros((400, 430, 1))  # 广州
        f.seek(0)
        # raw data
        # RawData = np.array([int(i) for i in f.read()])
        RawData = np.array([i for i in f.read()])

        RawData[RawData > 80] = 0

        try:
            RawArray = RawData.reshape(400, 400) if args.name == 'bj' else RawData.reshape(400, 430)  # 北京
            # RawArray = RawData.reshape(400, 430)  # 广州
        except:
            RawArray = np.zeros((400, 400)) if args.name == 'bj' else np.zeros((400, 430))  # 北京
            # RawArray = np.zeros((400, 430))
        img[:, :, 0] = RawArray
        if not os.path.exists(output):
            os.mkdir(output)
        imgpath = output + '/' + str(file) + '.png'
        # img[img > 58] = 0
        img[img > 0] += 32
        img[img > 0] *= 2
        # img[img > 110] = 100
        cv2.imwrite(imgpath, img)
        count += 1

    print('\n%d files' % count)


def parse_args():
    parser = argparse.ArgumentParser(description='parse BJ and GZ')
    parser.add_argument('--name', help="",
                        default='bj', type=str)
    parser.add_argument('--input', help="",
                        default='/data1/data2/wxyt/0709气象数据/2014', type=str)
    parser.add_argument('--output', help="",
                        default='/data1/data2/wxyt/0709气象数据/BJ_2014_pic', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # unbz2(args.input)

    parseBin(args.input, args.output)
    # 读完之后删除中间文件
    print('解析完成！')
