import cv2
from PIL import Image
import numpy as np
from paddleseg.core import predict, predict_single
from seg_utils import SegUtils
from paddleseg.utils import utils
from paddleseg.transforms import Compose
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg import utils
import time
import os
target_dir = os.path.join(os.path.dirname(__file__), "detect/yolov5-6.2")
from models.common import DetectMultiBackend

from utils.torch_utils import select_device, smart_inference_mode, time_sync



                
def test_single(segUtils,model_seg, model_det, device, src_img):
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
    pred = segUtils.remove_small_white_regions(pred*255, min_area=50000)
    t3 = time.time()
    print(f"remove_small_white_regions time: {t3 - t2:.4f} s")
    added_image = None
    # added_image = visualize_item(
    #     src_img, pred,  weight=0.2)
    t4 = time.time()
    print(f"visualize_item time: {t4 - t3:.4f} s")

    borders = segUtils.crop_detect_region(pred)
    if len(borders) == 0:
        return None,None
    t5 = time.time()
    print(f"crop_detect_region time: {t5 - t4:.4f} s")
    left_points, right_points, left_lane_region, right_lane_region,middle_lane_region = segUtils.get_Emergency_Lane(pred)
    t6 = time.time()
    print(f"get_Emergency_Lane time: {t6 - t5:.4f} s")
    detect_region = src_img[borders[0]:borders[1], borders[2]:borders[3]]
    detect_region = np.ascontiguousarray(detect_region)
    det_result = segUtils.detect_car(model_det, detect_region, device)
    t7 = time.time()
    print(f"detect_car time: {t7 - t6:.4f} s")
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
    # cv2.namedWindow("added_image", cv2.WINDOW_NORMAL)
    # cv2.imshow("detect_region", detect_region)
    # cv2.imshow("added_image", added_image)
    cv2.imshow("marked_points", show_img)
    # cv2.imshow("mask", mask)
    # cv2.imshow("final_mask", final_mask)
    # cv2.waitKey(0)
    return added_image, show_img


if __name__ == "__main__":
    """
    1. 分割
    2. 去除小白色区域
    3. 裁剪检测区域、获取应急车道线
    4. 检测车辆
    5. 判断车辆是否在应急车道线左侧
    """
    segUtils = SegUtils()
    
    cfg = Config("configs/quick_start/pp_liteseg_optic_disc_512x512_1k.yml")
    test_config = cfg.test_config
    builder = SegBuilder(cfg)
    utils.set_device("gpu:0")
    model_seg = builder.model
    transforms = Compose(builder.val_transforms)
    model_path = "output/best_model/model.pdparams"
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
    video_path = "/home/ubuntu/Desktop/DJI_20250501091406_0001.mp4"  # 替换为你的视频路径
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    # 获取视频参数，初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out_video_path = "/home/ubuntu/Desktop/DJI_20250501091406_0001_out.mp4"
    out_video_path_lane = "/home/ubuntu/Desktop/DJI_20250501091406_0001_out_lane.mp4"
    # out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    out_lane = cv2.VideoWriter(out_video_path_lane, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        temp_img_path = f"temp_frame_{frame_id}.jpg"
        frame_id += 1

        
        print(f"Processing video frame: {frame_id}")
        # test_single 处理后返回 show_img（你可以让 test_single 返回 show_img）
        seg_img,lane_img = test_single(segUtils,model_seg, model_det, device, frame)
        # cv2.imwrite(temp_img_path, show_img)

        # 写入视频
        # out.write(seg_img)
        out_lane.write(lane_img)
        # 按q键中断
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video interrupted by user.")
            break

    cap.release()
    # out.release()
    out_lane.release()
