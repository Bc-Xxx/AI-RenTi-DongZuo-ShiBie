# -*- coding: utf-8 -*-
"""
抛掷检测演示：证据①(人在布防区) + 证据③(挥臂动作)
用法：
  1. 拍一段测试视频(手机即可，10秒)：人走动、正常挥手、快速抛掷动作
  2. 改下面的 SOURCE 为视频文件名(或 0 用 USB 摄像头)
  3. 运行: python throw_demo.py
标定方法：看终端打印的"手腕速度"——正常挥手和真抛掷的数值会差几倍，
         把 SPIKE 调到两者之间
"""
import cv2
import math
import numpy as np
from ultralytics import YOLO

# ========== 配置区(现场标定改这里) ==========
SOURCE = 0               # 0 = USB/内置摄像头；测试视频填文件名如 "test.mp4"；现场相机填 "rtsp://用户名:密码@相机IP:554/..."
SPIKE = 40               # 手腕速度阈值(px/帧)，超过视为快速挥臂
COOLDOWN = 50            # 触发一次后的冷却帧数，防止连续刷屏
# 布防区多边形(画面里围栏内侧区域)。
# 演示阶段设为 None = 自动用整幅画面当布防区(适配任意分辨率的测试视频)
# 现场部署时改成围栏内侧实际标定的多边形坐标，例如：
# ZONE = np.array([[100, 1080], [100, 600], [900, 500], [1800, 600], [1800, 1080]], np.int32)
ZONE = None
# ============================================

WRIST, SHOUL = 10, 6     # COCO关键点编号：右腕、右肩

det = YOLO("yolo11s.pt")
pose = YOLO("yolo11s-pose.pt")

prev_wrist = {}   # 每个追踪ID上一帧的手腕位置
cooldown = 0

def in_zone(cx, cy):
    return cv2.pointPolygonTest(ZONE, (float(cx), float(cy)), False) >= 0

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise SystemExit(f"打不开视频源: {SOURCE} —— 检查文件名，或改成 0 用摄像头")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if ZONE is None:   # 首帧自动按视频分辨率生成布防区(演示模式=整幅画面)
        fh, fw = frame.shape[:2]
        ZONE = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]], np.int32)

    # 证据①：检测人，只看布防区内的
    result = det(frame, classes=[0], imgsz=480, verbose=False)[0]
    alert = False
    speed_labels = []   # 本帧所有人的(框左上x, y, 速度, 是否高于肩)
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) / 2, y2          # 脚底中心 = 人的位置
        if not in_zone(cx, cy):
            continue

        # 证据③：只对布防区内的人裁小图跑姿态模型
        crop = frame[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            continue
        pr = pose(crop, imgsz=320, verbose=False)[0]
        if pr.keypoints is None or len(pr.keypoints) == 0:
            continue
        kpt = pr.keypoints.xy[0]
        wrist, sh = kpt[WRIST], kpt[SHOUL]
        if wrist[0] <= 0 or sh[0] <= 0:     # 关键点没检出来
            continue

        # 区分人：demo用脚底横坐标分桶；正式版换 det.track() 拿真实ID
        pid = int(cx) // 60
        speed = 0 if pid not in prev_wrist else math.dist(wrist, prev_wrist[pid])
        prev_wrist[pid] = wrist

        above = wrist[1] < sh[1]            # 手腕高于肩线(y越小越高)
        speed_labels.append((x1, y1, speed, above))

        if speed > SPIKE and above and cooldown <= 0:
            print(">>> [证据③命中] 布防区内快速挥臂，疑似抛掷动作！")
            alert = True
            cooldown = COOLDOWN

    cooldown -= 1

    # 画布防区
    cv2.polylines(frame, [ZONE], True, (0, 255, 255), 2)
    if alert:
        cv2.putText(frame, "THROW ACTION!", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    # 画检测框和骨架(整帧可视化)
    plotted = result.plot()
    # 每个人头顶显示实时手腕速度：红色=手腕高于肩，绿色=低于肩
    for sx, sy, sp, ab in speed_labels:
        cv2.putText(plotted, f"{sp:.0f}", (sx, max(sy - 8, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255) if ab else (0, 255, 0), 2)
    plotted = cv2.polylines(plotted, [ZONE], True, (0, 255, 255), 2)
    if alert:
        cv2.putText(plotted, "THROW ACTION!", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imshow("throw_demo (ESC退出)", plotted)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
