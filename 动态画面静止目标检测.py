import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

class DynamicStaticDetector:
    def __init__(self, yolo_model='yolov5l6.pt', static_thres=1, min_static_frames=1):
        # 指定使用cuda
        self.detector = YOLO(yolo_model)
        self.detector.to('cuda')
        self.track_history = defaultdict(list)  # {id: [中心点]}
        self.static_count = defaultdict(int)    # {id: 静止帧计数}
        self.STATIC_THRES = static_thres        # 光流平均位移阈值（像素）
        self.MIN_STATIC_FRAMES = min_static_frames
        self.prev_frame = None                  # 前一帧（用于光流）
        self.prev_gray = None                   # 前一帧灰度图

    def _compensate_motion(self, prev_gray, curr_gray):
        # ORB特征点匹配，估算单应性矩阵，补偿相机运动
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return curr_gray
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return curr_gray
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is not None:
            return cv2.warpPerspective(curr_gray, H, (curr_gray.shape[1], curr_gray.shape[0]))
        return curr_gray

    def _mean_optical_flow(self, prev_gray, curr_gray, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi_prev = prev_gray[y1:y2, x1:x2]
        roi_curr = curr_gray[y1:y2, x1:x2]
        if roi_prev.size == 0 or roi_curr.size == 0:
            return 0
        p0 = cv2.goodFeaturesToTrack(roi_prev, maxCorners=30, qualityLevel=0.01, minDistance=3)
        if p0 is None or len(p0) == 0:
            return 0
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_prev, roi_curr, p0, None, winSize=(15,15), maxLevel=2,
                                               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        if p1 is None or st is None:
            return 0
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        if len(good_new) == 0:
            return 0
        displacements = np.linalg.norm(good_new - good_old, axis=1)
        mean_disp = np.mean(displacements)
        return mean_disp

    def process_frame(self, frame):
        # Step 1: 灰度图
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = curr_gray.copy()
            self.prev_frame = frame.copy()
            return frame

        # Step 2: 运动补偿，获得对齐后的当前帧灰度图
        compensated_gray = self._compensate_motion(self.prev_gray, curr_gray)

        # Step 3: YOLOv8检测+自带跟踪（用cuda）
        classes = [2,5,7]
        results = self.detector.track(frame, persist=True, verbose=False,imgsz=1280, classes=classes, device='cuda')[0]
        if hasattr(results, "boxes") and hasattr(results.boxes, "id") and results.boxes.id is not None:
            ids = results.boxes.id.int().cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
        else:
            ids = []
            boxes = []

        # Step 4: 静止点检测（用光流，基于补偿后帧）
        current_ids = set()
        for i, bbox in enumerate(boxes):
            track_id = int(ids[i]) if i < len(ids) else i
            mean_disp = self._mean_optical_flow(self.prev_gray, compensated_gray, bbox)
            current_ids.add(track_id)
            if mean_disp < self.STATIC_THRES:
                self.static_count[track_id] += 1
            else:
                self.static_count[track_id] = 0
            color = (0, 0, 255) if self.static_count[track_id] > self.MIN_STATIC_FRAMES else (0, 255, 0)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, f"ID:{track_id} disp:{mean_disp:.2f}", (int(bbox[0]), int(bbox[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 清理消失的目标
        for tid in list(self.static_count.keys()):
            if tid not in current_ids:
                del self.static_count[tid]
        self.prev_gray = curr_gray.copy()
        self.prev_frame = frame.copy()
        return frame

# 主函数
if __name__ == "__main__":
    detector = DynamicStaticDetector()
    cap = cv2.VideoCapture("F:/data/gaokong/Drone/DJI_20250501091754_0003_V.MP4")  # 替换为动态背景视频

    # 新增：初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("F:/data/gaokong/Drone/output/DJI_20250501091754_0003_V.MP4", fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        processed_frame = detector.process_frame(frame)
        cv2.namedWindow("Dynamic Static Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Dynamic Static Detection", processed_frame)
        out.write(processed_frame)  # 写入到新视频文件
        if cv2.waitKey(1) == ord('q'): break
    cap.release()
    out.release()  # 释放写入对
