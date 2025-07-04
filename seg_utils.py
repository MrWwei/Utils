import cv2
import numpy as np
from paddleseg.core import predict, predict_single
import os
import sys
target_dir = os.path.join(os.path.dirname(__file__), "detect/yolov5-6.2")
print(f"Adding {target_dir} to sys.path")
sys.path.append(target_dir)
import torch
import torch.backends.cudnn as cudnn

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                        increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.augmentations import letterbox



class SegUtils:
    def __init__(self):
        pass

    def remove_small_white_regions(self, mask, min_area=500):
        """
        填补大空洞，并去除小于min_area的白色区域
        """
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        def fill_holes(img):
            floodfill = img.copy()
            h, w = img.shape[:2]
            mask_ff = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(floodfill, mask_ff, (0,0), 255)
            cv2.floodFill(floodfill, mask_ff, (w-1,0), 255)
            cv2.floodFill(floodfill, mask_ff, (0,h-1), 255)
            cv2.floodFill(floodfill, mask_ff, (w-1,h-1), 255)
            return cv2.bitwise_not(floodfill)

        holes_mask = fill_holes(closed)
        filled = cv2.bitwise_or(closed, holes_mask)

        _, binary_img = cv2.threshold(filled, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(filled)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.drawContours(final_mask, [contour], -1, 255, -1)
        return final_mask

    def visualize_item(self, image, result, weight=0.6):
        """
        Convert predict result to color image, and save added image.
        """
        color_map = [0, 255, 255] * 256
        color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
        color_map = np.array(color_map).astype("uint8")

        im = cv2.imread(image)

        c1 = cv2.LUT(result, color_map[:, 0])
        c2 = cv2.LUT(result, color_map[:, 1])
        c3 = cv2.LUT(result, color_map[:, 2])

        pseudo_img = np.dstack((c3, c2, c1))

        mask = (result > 0).astype(np.uint8)
        mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        vis_result = im.copy()
        blended = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)
        vis_result[mask_3c == 1] = blended[mask_3c == 1]
        return vis_result

    def get_Emergency_Lane(self, mask):
        # 根据分割图得到双向的应急车道线，并返回中间区域
        left_border_points = []
        right_border_points = []
        left_quarter_points = []
        right_quarter_points = []

        p_interval = 7/42

        height, width = mask.shape
        row = mask[height-1, :]
        white_indices = np.where(row == 255)[0]
        start_col = white_indices[0]
        end_col = white_indices[-1]
        for y in range(height):
            row = mask[y, :]
            white_indices = np.where(row == 255)[0]
            if len(white_indices) > 0:
                start_col = white_indices[0]
                end_col = white_indices[-1]
                left_border_points.append((start_col, y))
                right_border_points.append((end_col, y))
                left_quarter_col = start_col + int((end_col - start_col) * p_interval)
                left_quarter_points.append((left_quarter_col, y))
                right_quarter_col = end_col - int((end_col - start_col) * p_interval)
                right_quarter_points.append((right_quarter_col, y))

        left_lane_region = left_border_points + left_quarter_points[::-1]
        right_lane_region = right_border_points + right_quarter_points[::-1]
        middle_lane_region = left_quarter_points + right_quarter_points[::-1]
        return left_quarter_points, right_quarter_points, left_lane_region, right_lane_region, middle_lane_region

    def crop_detect_region(self, mask_thre):
        img = mask_thre
        white_pixels = np.column_stack(np.where(img > 0))
        y_coords = white_pixels[:, 0]
        if y_coords.size == 0:
            return ()
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        white_height = y_max - y_min
        y_start = y_min + int(white_height * 1/5)
        y_end = y_max
        mask = (white_pixels[:, 0] >= y_start) & (white_pixels[:, 0] <= y_end)
        retained_pixels = white_pixels[mask]
        x_min, x_max = np.min(retained_pixels[:, 1]), np.max(retained_pixels[:, 1])
        y_min_new, y_max_new = np.min(retained_pixels[:, 0]), np.max(retained_pixels[:, 0])
        x1 = max(0, x_min - 1)
        y1 = max(0, y_min_new - 1)
        x2 = min(img.shape[1], x_max + 1)
        y2 = min(img.shape[0], y_max_new + 1)
        return (y1, y2, x1, x2)

    def detect_car(self, model, img0, device):
        img = letterbox(img0, 640, stride=False, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device).unsqueeze(0)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = model(im, augment=False, visualize=False)
        classes = None
        conf_thres = 0.6
        iou_thres = 0.45
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=1000)
        for i, det in enumerate(pred):
            
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                return det
