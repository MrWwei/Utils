import cv2
import numpy as np
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', type=str, default='label2coco',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--output', type=str, default='label2coco',
                        help='name of the experiment. It decides where to store samples and models')
    args = parser.parse_args()

    for img_path in os.listdir(args.input):
        img = cv2.imread(os.path.join(args.input, img_path))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite(os.path.join(args.output, img_path))
