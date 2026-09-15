# -*- coding: utf-8 -*-
"""
RTSP 实时监控测试(低延迟版)：
  后台线程持续收帧、只保留最新一帧，主循环永远处理"此刻"的画面，
  避免 CPU 推理慢导致旧帧积压、画面越来越滞后。
操作：
  - 画面显示 检测框+骨架+头顶手腕速度(红=高于肩,绿=低于肩) 和处理帧率FPS
  - 按 s 保存快照 snapshot_N.jpg   按 ESC 退出
"""
import os
import time
import math
import threading
import csv
import datetime
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"  # 强制TCP防丢包

RTSP = "rtsp://<用户名>:<密码>@<相机IP>:554/Streaming/Channels/301"
SPIKE = 40
COOLDOWN = 50
WRIST, SHOUL = 10, 6


class LiveCapture:
    """后台线程收帧，只保留最新帧；断流自动重连"""

    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                print("画面中断，尝试重连...")
                self.cap.release()
                time.sleep(0.5)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                continue
            with self.lock:
                self.frame = f        # 旧帧直接被覆盖 = 不积压

    def read(self):
        with self.lock:
            f = self.frame
        return f                      # None 表示还没收到画面

    def release(self):
        self.running = False
        time.sleep(0.2)
        self.cap.release()


det = YOLO("yolo11s.pt")
pose = YOLO("yolo11s-pose.pt")

prev_wrist = {}
peak_hist = {}          # 每人最近N帧的速度窗口，屏幕显示窗口峰值(数字保持可读)
cooldown = 0
snap_id = 0
ZONE = None
fps, t_prev = 0.0, time.time()
log = csv.writer(open("calibration_log.csv", "a", newline="", encoding="utf-8"))
log.writerow(["时间", "人ID", "手腕速度px每帧", "高于肩"])   # 标定数据全部落盘

def in_zone(cx, cy):
    return cv2.pointPolygonTest(ZONE, (float(cx), float(cy)), False) >= 0

print(f"正在连接相机: {RTSP}")
cam = LiveCapture(RTSP)

while True:
    frame = cam.read()
    if frame is None:
        cv2.waitKey(100)
        continue
    if ZONE is None:
        fh, fw = frame.shape[:2]
        ZONE = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]], np.int32)

    result = det(frame, classes=[0], imgsz=480, verbose=False)[0]
    alert = False
    speed_labels = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) / 2, y2
        if not in_zone(cx, cy):
            continue
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            continue
        pr = pose(crop, imgsz=320, verbose=False)[0]
        if pr.keypoints is None or len(pr.keypoints) == 0:
            continue
        kpt = pr.keypoints.xy[0]
        wrist, sh = kpt[WRIST], kpt[SHOUL]
        if wrist[0] <= 0 or sh[0] <= 0:
            continue
        pid = int(cx) // 60
        speed = 0 if pid not in prev_wrist else math.dist(wrist, prev_wrist[pid])
        prev_wrist[pid] = wrist
        above = wrist[1] < sh[1]
        win = peak_hist.setdefault(pid, deque(maxlen=12))   # 约3~5秒的峰值窗口
        win.append(speed)
        disp = max(win)                                     # 显示峰值,数字不闪
        speed_labels.append((x1, y1, disp, above))
        log.writerow([datetime.datetime.now().strftime("%H:%M:%S"),
                      pid, f"{speed:.0f}", above])
        if speed > SPIKE and above and cooldown <= 0:
            print(">>> [证据③命中] 布防区内快速挥臂，疑似抛掷动作！")
            alert = True
            cooldown = COOLDOWN
    cooldown -= 1

    plotted = result.plot()
    for sx, sy, sp, ab in speed_labels:
        cv2.putText(plotted, f"{sp:.0f}", (sx, max(sy - 8, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255) if ab else (0, 255, 0), 2)
    plotted = cv2.polylines(plotted, [ZONE], True, (0, 255, 255), 2)
    if alert:
        cv2.putText(plotted, "THROW ACTION!", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(plotted, f"FPS:{fps:.1f}", (plotted.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.imshow("rtsp_test (s=快照 ESC=退出)", plotted)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))   # 平滑显示处理帧率
    t_prev = now

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        snap_id += 1
        fn = f"snapshot_{snap_id}.jpg"
        cv2.imwrite(fn, frame)
        print(f"已保存快照 {fn}")
    elif key == 27:
        break

cam.release()
cv2.destroyAllWindows()
