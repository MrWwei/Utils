#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import struct
import threading
import multiprocessing
import queue 
import numpy as np
import tqdm
import zipfile
import cv2
import bz2


def touch_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def read_cref(queue):
    # latlon = open(FileName, "rb")
    # latlon = bz2.open(FileName, 'r')
    while True:
        deq = queue.get()
        if deq is None: break
        File, target_path = deq
        latlon = bz2.BZ2File(File)

        # 1++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # latlon.seek(128, 1)
        # latlon.seek(32, 1)
        # latlon.seek(16, 1)
        # latlon.seek(2, 1)
        # latlon.seek(2, 1)
        # slat = struct.unpack("f", latlon.read(4))[0]
        # Wlon = struct.unpack("f", latlon.read(4))[0]
        # Nlat = struct.unpack("f", latlon.read(4))[0]
        # elon = struct.unpack("f", latlon.read(4))[0]
        # clat = struct.unpack("f", latlon.read(4))[0]
        # clon = struct.unpack("f", latlon.read(4))[0]
        # ----------------
        latlon.seek(204, 1)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        rows = struct.unpack("i", latlon.read(4))[0]
        cols = struct.unpack("i", latlon.read(4))[0]

        # 2++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # dkat = struct.unpack("f", latlon.read(4))[0]
        # dlon = struct.unpack("f", latlon.read(4))[0]
        # ----------------
        latlon.seek(8, 1)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        nodata = struct.unpack("f", latlon.read(4))[0]

        # 3++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # levelbytes = struct.unpack("i", latlon.read(4))[0]
        # levelnum = struct.unpack("h", latlon.read(2))[0]
        # ----------------
        latlon.seek(6, 1)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        amp = struct.unpack("h", latlon.read(2))[0]

        # 4++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # compmode = struct.unpack("h", latlon.read(2))[0]
        # dates = struct.unpack("h", latlon.read(2))[0]
        # seconds = struct.unpack("i", latlon.read(4))[0]
        # min_value = struct.unpack("h", latlon.read(2))[0]
        # max_value = struct.unpack("h", latlon.read(2))[0]
        # ----------------
        latlon.seek(12, 1)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        latlon.seek(12, 1)
        y = struct.unpack("h", latlon.read(2))[0]
        x = struct.unpack("h", latlon.read(2))[0]
        n = struct.unpack("h", latlon.read(2))[0]
        ref_data = np.ones((int(rows), int(cols))) * nodata
        count = 0
        while y != -1 or x != -1 and n != -1:
            tempdata = struct.unpack("{}h".format(int(n)), latlon.read(int(n) * 2))
            ref_data[y, (x):x + n] = np.array(tempdata) / amp
            y = struct.unpack("h", latlon.read(2))[0]
            x = struct.unpack("h", latlon.read(2))[0]
            n = struct.unpack("h", latlon.read(2))[0]
            count += n
        # print(count)
        latlon.close()
        ref_data[np.where(ref_data <= 0)] = 0
        ref_data = cv2.flip(ref_data, 0).astype(np.uint8)
        cv2.imwrite(target_path, ref_data)
        latlon.close()


def unzip_read(tar_g):
    zip_ = zipfile.ZipFile(tar_g)
    zip_.testzip()
    names = zip_.namelist()
    print("\n", tar_g, "NUM: ", len(names), "\n")
    for name in names:
        if name.endswith("bz2"):
            f = zip_.open(name)
            yield f, name
    zip_.close()


def main(num_threads, target_path, root_path):
    q_list = [queue.Queue(16) for _ in range(num_threads)]
    p_list = [threading.Thread(target=read_cref, args=(q_list[i],)) for i in range(num_threads)]
    for p in p_list: p.start()

    for i in ["2016", "2017", "2018"]:
        touch_dir(os.path.join(target_path, i))
    count = 0
    for zip_d in [i for i in os.listdir(root_path) if i.endswith(".zip")]:
        file_list = tqdm.tqdm(unzip_read(os.path.join(root_path, zip_d)))
        for file1, name in file_list:
            file_list.set_description(name)
            tar = os.path.join(target_path, name[:-4] + ".png")
            if os.path.exists(tar): continue
            q_ind = count % num_threads
            while q_list[q_ind].qsize() == num_threads:
                q_ind += 1
                if q_ind == num_threads: q_ind = 0

            # read_cref(file1, tar)
            q_list[q_ind].put((file1, tar))
            count += 1

    for q in q_list: q.put(None)
    for p in p_list: p.join()


if __name__ == '__main__':
    target_path = '/data2/wxyt/radar_puzzle'
    root_path = "/data2/wxyt/radar_puzzle"
    num_threads = 4 
    # read_cref(d, tar_path)

    main(num_threads, target_path, root_path)

