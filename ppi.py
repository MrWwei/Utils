#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import wradlib as wrl
import numpy as np
import os
from wradlib.io import iris
import cv2
import math
import argparse
import tqdm
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='wxyt ppi2cappi data preprocess')
    # general
    parser.add_argument('--root_dir', default="/home/em/桌面/radardata", help='')
    parser.add_argument('--save_dir', default="/home/em/桌面/radardata", help='')
    parser.add_argument('--height', default=7, type=float, help='Max of A is 11.7, B is 30')
    # parser.add_argument('--product_type',无标题文档default='VOL_A',help='')
    args = parser.parse_args()
    # print(args)
    return args


'''
直角坐标转换为球坐标,再转换为体扫数据格式
'''


def xyz2ppi(x, y, z, product_type):
    if product_type == 'VOL_A':
        r0 = 830.0/112
    elif product_type == 'VOL_B':
        r0 = 414.0/112
    else:
        print('product type error! It must be VOL_A or VOL_B.')
        assert False
    if x == 0:
        x = 0.0000001
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.pi/2-math.acos(z*1.0/r)
    phi = math.atan(y*1.0/x)
    n = int(r*r0)
    m = int(180*phi/math.pi)
    if x < 0:
        m += 180
    if (x > 0) and (y < 0):
        m = 360 + m
    return m, n, theta


'''无标题文档
插值得到等高面数据
'''


def interpolation(m, n, theta, ppi_nsweep, angle_theta):
    theta_diff = []
    # print (angle_theta)
    theta_diff[:] = [abs(x - theta) for x in angle_theta]
    theta_diff_min = min(theta_diff)
    theta_diff_min_index = theta_diff.index(theta_diff_min)
    ppi = ppi_nsweep[:, :, theta_diff_min_index]
    epsilon = 10
    D = float('inf')
    dbz = 0
    print('test')
    for i in range(max(0, m-epsilon), min(ppi.shape[0], m+epsilon), 1):
        for j in range(max(0, n-epsilon), min(ppi.shape[1], n+epsilon), 1):
            dist = math.sqrt((i-m)**2 + (j-n)**2)
            if dist < D:
                D = dist
                dbz = ppi[i, j]
    return dbz


'''
生成等高面数据
'''


def ppi2cappi(height, ppi_nsweep, angle_theta, product_type):
    # fix l and w by default
    lenth = width = 224
    cappi = np.zeros(shape=(224,224))
    for x in range(lenth):
        x -= 112
        for y in range(width):
            y -= 112
            m, n, theta = xyz2ppi(x, y, height, product_type)
            dbz = interpolation(m, n, theta, ppi_nsweep, angle_theta)
            cappi[x+112][y+112] = dbz   #无标题文档
    return cappi


'''
确定产品型号
'''


def check_parse_iris(file_name):
    irisfile = iris.IrisFile(file_name)
    header = irisfile.product_hdr["product_configuration"]
    product_type = header["product_name"].strip()
    if product_type == 'VOL_A' or product_type == 'VOL_B':
  #  if product_type == 'VOL_A':
        return True, product_type

    # if product_type_actual == product_type_given:
    #     return True, product_type_actual
    else:
        return False, None


'''
#从原始文件中提取体扫数据
'''


def extract(i):
    flag, product_type = check_parse_iris(i)
    print(product_type)
    # 确定产品型号为VOL_A 或 VOL_B
    if flag:
        # assert (product_type != None)
        # 获得体扫数据

        fcontent2, nsweep = wrl.io.read_iris(i)
        angle_theta = []
        if product_type == 'VOL_A':
            ppi_nsweep = np.zeros((360, 831, 3))
        elif product_type == 'VOL_B':  #无标题文档
            ppi_nsweep = np.zeros((360, 414, 5))
        # VOL_A 3个角度, VOL_B 5个角度
        for sweep in range(nsweep):
            data = fcontent2["data"][sweep + 1]
            for k, v in data['ingest_data_hdrs'].items():
                if k == 'DB_XHDR':
                    for k1, v1 in v.items():
                        if k1 == 'fixed_angle':
                            # theta = v1 / 180 * math.pi
                            # 获取仰角theta,保留小数点后一位.
                            angle_theta.append(round(v1, 1))
            for key, value in data['sweep_data'].items():
                if key == 'DB_DBZ':
                    value['data'] += 32
                    value['data'] *= 2
                    ppi_sweep = value['data']
                    ppi_sweep = np.asarray(ppi_sweep)
                    # ppi_sweep = np.expand_dims(ppi_sweep,axis=2)
                    ppi_nsweep[:, :, sweep] = ppi_sweep
        return ppi_nsweep, angle_theta, product_type
    else:
        return None, None, None

import numpy.ma as ma
import matplotlib.pyplot as plt
def extractvel(i):
    fcontent2 = wrl.io.read_iris(i)
    ftype = fcontent2['product_hdr']['product_configuration']["product_name"].strip()
    for _, data in fcontent2['data'].items():
        vel = data["sweep_data"]["DB_VEL"]
        vel_data = vel['data']
        ma.set_fill_value(vel_data, 0)
        vel_data = vel_data.filled()
        vel_angel = vel['ele_start'][0]
        break
    cdict = ['#E6E6E6', '#00E0FE', '#0080FF', '#320096', '#00FB90',
             '#00BB99', '#008F00', '#CDC99F', '#767676', '#F88700', '#FFCF00', '#FFFF00',
             '#AE0000', '#D07000', '#FF0000', '#FF007D']
    cmap = colors.ListedColormap(cdict)
    norm = colors.Normalize(vmin=-50, vmax=50)
    bins = 778
    if ftype == 'VOL_A':
        bins = 831
    elif ftype == 'VOL_B':  # 无标题文档
        bins = 414
    el = np.zeros((360, bins))  # 仰角
    for i in range(360):
        for j in range(bins):
            el[i, j] = vel_angel
    az = np.zeros((360, bins))  # 方位角
    for i in range(360):
        for j in range(bins):
            az[i, j] = i
    rl = np.zeros((360, bins))  # 径向长度
    for i in range(360):
        for j in range(bins):
            rl[i, j] = j
#    dbz = np.zeros((len(data), 460))  # 反射率
    x, y, h = sph2cord(el, az, rl)
    x = np.concatenate((x, [x[0]]))  # 闭合
    y = np.concatenate((y, [y[0]]))  # 闭合
    plt.pcolor(x, y, vel_data, norm=norm, cmap=cmap)
    plt.title('Velocity')
    plt.axis('square')
    plt.colorbar()
    plt.show()
'''''
    fig = plt.gcf()
    fig.set_size_inches(720 / 100, 720 / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax, pm = wrl.vis.plot_ppi(vel_data, fig=fig)
    plt.axis('off')
    plt.show()
    break
    print(ftype)
'''''

from matplotlib import colors
def extractref(i):
    picname = i
    fcontent2 = wrl.io.read_iris(i)
    ftype = fcontent2['product_hdr']['product_configuration']["product_name"].strip()
    elevation = 0
    if ftype == 'VOL_A':
        bins = 831
        # elnum =
    elif ftype == 'VOL_B':  # 无标题文档
        bins = 414
    elif ftype == 'WIND':
        bins = 778
    dbz_data = np.zeros((8, 360, bins), dtype=np.float_)
    dbz_data = np.full(dbz_data.shape, -32)
    dbz_angel = np.zeros(fcontent2['nsweeps'], dtype=np.float)  # 仰角
    for _, data in fcontent2['data'].items():
        dbz = data["sweep_data"]["DB_DBZ"]
        dbz_data[elevation] = dbz['data']
        dbz_angel[elevation] = dbz['ele_start'][elevation]
        elevation += 1


    return dbz_data, dbz_angel, bins
    '''''
    fig = plt.gcf()
    fig.set_size_inches(720 / 100, 720 / 100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax, pm = wrl.vis.plot_ppi(dbz_data, fig=fig)
    plt.axis('off')
    plt.show()
'''''






def sph2cord(el, az, r):
    e, a = np.deg2rad([el, az])
    x = r * np.cos(e) * np.sin(a)
    y = r * np.cos(e) * np.cos(a)
    h = r * np.sin(e)
    return x, y, h
def drawpic(data, productid, pathname):
    cdict = ['#606060', '#01ADA5', '#C0C0FE', '#7B72EF', '#1F27D1',
             '#A6FDA8', '#00EA00', '#10921A', '#FCF465', '#C9C903', '#8C8C00',
             '#FFACAC', '#FE655C', '#EE0231', '#D58FFE', '#AA25FA', '#FFFFFF']
    cmap = colors.ListedColormap(cdict)
    if productid == 1 or productid == 2 or productid == 3:
        norm = colors.Normalize(vmin=-10, vmax=70)
    elif productid == 4:
        norm = colors.Normalize(vmin=0, vmax=25)
    picname = ''
    el = np.zeros((360, bins))  # 仰角
    for i in range(360):
        for j in range(bins):
            el[i, j] = dbz_angel[0]
    az = np.zeros((360, bins))  # 方位角
    for i in range(360):
        for j in range(bins):
            az[i, j] = i
    rl = np.zeros((360, bins))  # 径向长度
    for i in range(360):
        for j in range(bins):
            rl[i, j] = j / 2
    #    dbz = np.zeros((len(data), 460))  # 反射率
    x, y, h = sph2cord(el, az, rl)
    # import pdb; pdb.set_trace()
    x = np.concatenate((x, [x[0]]))  # 闭合
    y = np.concatenate((y, [y[0]]))  # 闭合
    # np.ones(x)
    plt.pcolor(x, y, data, norm=norm, cmap=cmap)
    if productid == 1:
        productname = '-Ref'
    plt.title(productname)
    plt.axis('square')
    picname = pathname + productname + '.png'  # 分割，不带后缀名
    #os.path.splitext(picname)[0]
    plt.colorbar()
    plt.savefig(picname)
    plt.close(picname)
    plt.show()

def process():
    cnt = 0
    for f in os.listdir(args.root_dir):
        file = os.path.join(args.root_dir, f)
        file_list = [os.path.join(file, i) for i in os.listdir(file) if i.startswith("SHA")]
        for i in file_list:
            ppi_nsweep, angle_theta, product_type = extract(i)
            if ppi_nsweep is None or angle_theta is None:
                continue
            cappi = ppi2cappi(args.height, ppi_nsweep, angle_theta, product_type)
            # 垂直翻转
            cappi = cv2.flip(cappi, 0)
            img_save_root = args.save_dir.format(f)
            if not os.path.exists(img_save_root):
                os.mkdir(img_save_root)
            img_save_path = os.path.join(img_save_root, product_type)
            if not os.path.exists(img_save_path):
                os.mkdir(img_save_path)
            cv2.imwrite(os.path.join(img_save_path, '{}.png'.format(os.path.basename(i))), cappi)
            print('succeed:', cnt)
            cnt += 1

#无标题文档
def main():
    global args   #无标题文档
    args = parse_args()
    print(args)
    process()


if __name__ == '__main__':
    # main()
#    filename = "/data_1/雷达卫星原始数据/radar_zsss_raw_20170814/SHA170814042052.RAWF7VP"
#    extractvel(filename)
 #   extractref(filename)

    parser = argparse.ArgumentParser(description='s1、s2、s3， white to black')
    parser.add_argument('--input', type=str, default='/data1/yn', help='')
    args = parser.parse_args()
    # for root, dirs, files in os.walk(args.input):
    for file in tqdm.tqdm(os.listdir(args.input + '/data')):
        dbz_data, dbz_angel, bins = extractref(os.path.join(args.input + '/data', file))

        productid = 1
        file = os.path.splitext(file)[0]
        filefullname = args.input +'/pic/' +file
        drawpic(dbz_data[0], productid, filefullname)
