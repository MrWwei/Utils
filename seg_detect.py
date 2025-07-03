import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from paddleseg.core import predict, predict_single
def get_Emergency_Lane(mask, car_width=0):
    # 根据分割图得到双向的应急车道线
    # 新增：用于存储每一行的左边界点
    left_border_points = []
    right_border_points = []
    left_quarter_points = []
    right_quarter_points = []

    p_interval = 7/42

    row = mask[mask.shape[0]-1, :]
    white_indices = np.where(row == 255)[0]
    start_col = white_indices[0]
    end_col = white_indices[-1]
    for y in range(mask.shape[0]):
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
    # 新增：中间区域（左四分点+右四分点逆序）
    middle_lane_region = left_quarter_points + right_quarter_points[::-1]

    return left_quarter_points, right_quarter_points, left_lane_region, right_lane_region, middle_lane_region

def crop_detect_region(mask_thre):
    # 读取图像
    img = mask_thre
    
    # 确保是二值图像
    # _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    
    # 寻找所有白色像素坐标
    white_pixels = np.column_stack(np.where(img > 0))
    
    # 获取白色区域的垂直范围
    y_coords = white_pixels[:, 0]
    if y_coords.size == 0:
        return ()
    
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    white_height = y_max - y_min
    
    # 计算保留区域的下三分之二范围
    y_start = y_min + int(white_height * 1/3)
    y_end = y_max
    
    # 找出在保留区域内的白色像素
    mask = (white_pixels[:, 0] >= y_start) & (white_pixels[:, 0] <= y_end)
    retained_pixels = white_pixels[mask]
    
    # 计算新白色区域的最小外接矩形
    x_min, x_max = np.min(retained_pixels[:, 1]), np.max(retained_pixels[:, 1])
    y_min_new, y_max_new = np.min(retained_pixels[:, 0]), np.max(retained_pixels[:, 0])
    
    # 创建裁剪区域（扩展1像素确保完全包含边界）
    x1 = max(0, x_min - 1)
    y1 = max(0, y_min_new - 1)
    x2 = min(img.shape[1], x_max + 1)
    y2 = min(img.shape[0], y_max_new + 1)
    
    # 执行裁剪
    cropped = img[y1:y2, x1:x2]
    
    # 保存结果
    cv2.imwrite("detect_region_msk.jpg", cropped)
    return (y1, y2, x1, x2)
    # print(f"图像已保存至: {output_path}")

import numpy as np


import os
import sys
target_dir = os.path.join(os.path.dirname(__file__), "detect/yolov5-6.2")
print(f"Adding {target_dir} to sys.path")
sys.path.append(target_dir)
import torch
import torch.backends.cudnn as cudnn
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                        increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.augmentations import letterbox
from utils.torch_utils import select_device, smart_inference_mode, time_sync
from paddleseg.utils import logger, progbar, visualize
def remove_small_white_regions(mask, min_area=500):
    
    """
    填补大空洞，并去除小于min_area的白色区域
    """
    # 步骤1：闭运算填补小空洞
    kernel_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 步骤2：孔洞填充算法（处理闭运算未填充的大空洞）
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

    # 步骤3：去除小于min_area的白色区域
    _, binary_img = cv2.threshold(filled, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(filled)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(final_mask, [contour], -1, 255, -1)
    return final_mask
    
def visualize_item(im, result, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6
        use_multilabel (bool, optional): Whether to enable multilabel mode. Default: False.

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """
    # 生成全为黄色的 color_map（BGR: 0,255,255）
    color_map = [0, 255, 255] * 256
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")

    # im = cv2.imread(image)

    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])

    pseudo_img = np.dstack((c3, c2, c1))

    # 只在分割目标区域上叠加颜色
    mask = (result > 0).astype(np.uint8)
    mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    vis_result = im.copy()
    blended = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)
    vis_result[mask_3c == 1] = blended[mask_3c == 1]
    return vis_result
import time

def detect_car(model,img0,device):
    bs = 1  # batch_size
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    t1 = time_sync()
    img = letterbox(img0, 640, stride=False, auto=False)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).to(device).unsqueeze(0)  # uint8 to fp16/32, BHWC to BCHW
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im, augment=False, visualize=False)
    print("detector pred:")
    classes = None
    conf_thres = 0.6
    iou_thres = 0.45
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=1000)

    for i, det in enumerate(pred):  # per image
        seen += 1

        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()


            return det
                
def test_single(model_seg, model_det, device, src_img):
    """
    测试单个函数
    """
    # input_path = "mask.jpg"
    # src_img_path = "DJI_20250501091406_0001_V_frame_2060.jpg"
    # src_img = cv2.imread(src_img_path)
    # data = {}
    # data['img'] = src_img.astype('float32')
    show_img = src_img.copy()
    t0 = time.time()
    pred = predict_single(model=model_seg,
                                transforms=transforms,
                                img_path=src_img)
    t1 = time.time()
    print(f"predict_single time: {t1 - t0:.4f} s")
    t2 = time.time()
    pred = remove_small_white_regions(pred*255, min_area=50000)
    t3 = time.time()
    print(f"remove_small_white_regions time: {t3 - t2:.4f} s")
    
    added_image = visualize_item(
        src_img, pred,  weight=0.2)
    t4 = time.time()
    print(f"visualize_item time: {t4 - t3:.4f} s")

    borders = crop_detect_region(pred)
    if len(borders) == 0:
        return None,None
    t5 = time.time()
    print(f"crop_detect_region time: {t5 - t4:.4f} s")
    left_points, right_points, left_lane_region, right_lane_region,middle_lane_region = get_Emergency_Lane(pred)
    t6 = time.time()
    print(f"get_Emergency_Lane time: {t6 - t5:.4f} s")
    detect_region = src_img[borders[0]:borders[1], borders[2]:borders[3]]
    detect_region = np.ascontiguousarray(detect_region)
    # test_single(model_seg, model_det,device, image_path)
#
    det_result = detect_car(model_det, detect_region, device)
    if det_result is None:
        return None, None
    for *xyxy, conf, cls in reversed(det_result):
        x1 = xyxy[0] + borders[2]
        y1 = xyxy[1] + borders[0]
        x2 = xyxy[2] + borders[2]
        y2 = xyxy[3] + borders[0]
        # 计算中心的坐标
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x1 >= src_img.shape[1] or y1 >= src_img.shape[0] or x2 >= src_img.shape[1] or y2 >= src_img.shape[0]:
            continue
        color = (0,0,0)
        # 判断车辆是否在left_lane_region区域内
        if cv2.pointPolygonTest(np.array(left_lane_region, dtype=np.int32), (int(center_x), int(center_y)), False) >= 0:
            color = (0, 0, 255)  # 绿色表示在应急车道线左侧
        elif cv2.pointPolygonTest(np.array(right_lane_region, dtype=np.int32), (int(center_x), int(center_y)), False) >= 0:
            color = (0, 0, 255)
        elif cv2.pointPolygonTest(np.array(middle_lane_region, dtype=np.int32), (int(center_x), int(center_y)), False) >= 0:
            color = (0,255,0)
        else:
            continue
        cv2.rectangle(show_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.circle(show_img, (int(center_x), int(center_y)), 2, color, -1)
    for point_left, point_right in zip(left_points, right_points):
        # 画左侧和右侧的曲线
        if len(left_points) > 1:
            pts_left = np.array(left_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(show_img, [pts_left], isClosed=False, color=(0, 0, 255), thickness=2)
        if len(right_points) > 1:
            pts_right = np.array(right_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(show_img, [pts_right], isClosed=False, color=(0, 255, 0), thickness=2)
    # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("detect_region", cv2.WINDOW_NORMAL)
    cv2.namedWindow("marked_points", cv2.WINDOW_NORMAL)
    cv2.namedWindow("added_image", cv2.WINDOW_NORMAL)
    # cv2.imshow("detect_region", detect_region)
    cv2.imshow("added_image", added_image)
    cv2.imshow("marked_points", show_img)
    # cv2.imshow("mask", mask)
    # cv2.imshow("final_mask", final_mask)
    # cv2.waitKey(0)
    return added_image, show_img

from paddleseg.utils import get_image_list, get_sys_env, logger, utils
from paddleseg.transforms import Compose
from paddleseg.cvlibs import Config, SegBuilder, manager
from paddleseg import utils
import numpy as np
if __name__ == "__main__":
    """
    1. 分割
    2. 去除小白色区域
    3. 裁剪检测区域、获取应急车道线
    4. 检测车辆
    5. 判断车辆是否在应急车道线左侧
    """
    cfg = Config("configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k.yml")
    test_config = cfg.test_config
    builder = SegBuilder(cfg)
    utils.set_device("gpu:0")
    model_seg = builder.model
    transforms = Compose(builder.val_transforms)
    model_path = "output_big/best_model/model.pdparams"
    image_path = "dataset/images"
    # image_list, image_dir = get_image_list(image_path)
    utils.utils.load_entire_model(model_seg, model_path)
    model_seg.eval()

    device = '0'
    device = select_device(device)
    weights = "detect/CarGaoDian_TAG1_S640_V2.0.pt"  # 替换为您的模型权重路径
    model_det = DetectMultiBackend(weights, device=device, dnn=False, data='yolov5-6.2/data/coco128.yaml', fp16=False)
    # stride, names, pt = model.stride, model.names, model.pt
    # for image_path in image_list:
    #     print(f"Processing image: {image_path}")
    #     test_single(model_seg, model_det,device, image_path)
    
    # 处理视频
    video_path = "/home/ubuntu/Desktop/17_48_43-17_55_48-425000 00_01_02-00_03_20.mp4"  # 替换为你的视频路径
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    # 获取视频参数，初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video_path = "/home/ubuntu/Desktop/17_48_43-17_55_48-425000 00_01_02-00_03_20_out.mp4"
    out_video_path_lane = "/home/ubuntu/Desktop/17_48_43-17_55_48-425000 00_01_02-00_03_20_out_lane.mp4"
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    out_lane = cv2.VideoWriter(out_video_path_lane, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        temp_img_path = f"temp_frame_{frame_id}.jpg"
        frame_id += 1

        
        print(f"Processing video frame: {frame_id}")
        # test_single 处理后返回 show_img（你可以让 test_single 返回 show_img）
        seg_img,lane_img = test_single(model_seg, model_det, device, frame)
        # cv2.imwrite(temp_img_path, show_img)

        # 写入视频
        out.write(seg_img)
        out_lane.write(lane_img)
        # 按q键中断
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video interrupted by user.")
            break

    cap.release()
    out.release()
    out_lane.release()
