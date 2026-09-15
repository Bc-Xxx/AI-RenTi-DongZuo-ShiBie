# -*- coding: utf-8 -*-
"""
抽烟识别演示：抽烟目标检测(烟头/吸烟动作) + 人检测 + 关联判定
判定逻辑：抽烟模型的检出框 中心落在某个人的框内(且多帧确认) => 该人抽烟
操作：
  - 实时画面：蓝框=人，红框=抽烟目标，关联成功显示 SMOKING
  - 按 s 保存快照 snapshot_N.jpg
  - 按 ESC 退出
"""
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"  # 强制TCP防丢包

SOURCE = 0            # 0=摄像头；测试视频填 "test.mp4"；现场相机填 rtsp://...
SMOKE_MODEL = "smoke_best.pt"   # 抽烟检测权重(下载后放同目录)
CONFIRM_N = 8         # 连续N帧关联成功才告警(压误报的关键)
PERSON_CONF = 0.35    # 人检测置信度阈值
TARGET_CONF = 0.30    # 抽烟目标置信度阈值
MARGIN = 30           # 目标允许超出人框的像素余量(烟在手边可能稍微出框)

det = YOLO("yolo11s.pt")
smoke = YOLO(SMOKE_MODEL)
print("抽烟模型类别:", smoke.names)   # 第一次跑看一下类别名，发我确认

confirm = {}          # 每个人累计的关联帧数
snap_id = 0
ZONE = None
last_smoking = set()  # 上一帧在抽烟的人,用来判断"新发生"的告警

def in_zone(cx, cy):
    return cv2.pointPolygonTest(ZONE, (float(cx), float(cy)), False) >= 0

print(f"正在打开视频源: {SOURCE}")
backend = cv2.CAP_FFMPEG if isinstance(SOURCE, str) else cv2.CAP_DSHOW  # 网络流用FFMPEG,本地摄像头用DSHOW
cap = cv2.VideoCapture(SOURCE, backend)
if not cap.isOpened():
    raise SystemExit("打不开视频源")

while True:
    ok, frame = cap.read()
    if not ok:
        if isinstance(SOURCE, int):          # 摄像头偶发失败,重试
            cap.release(); cap = cv2.VideoCapture(SOURCE, cv2.CAP_DSHOW)
            continue
        break
    if ZONE is None:
        fh, fw = frame.shape[:2]
        ZONE = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]], np.int32)

    persons = det(frame, classes=[0], conf=PERSON_CONF, imgsz=480, verbose=False)[0]
    targets = smoke(frame, conf=TARGET_CONF, imgsz=640, verbose=False)[0]

    # 每个抽烟目标：找它属于哪个人(目标中心落在人框内+余量)
    pboxes = [list(map(int, b.xyxy[0])) for b in persons.boxes]
    smoking_ids = set()
    tboxes = []
    for tb in targets.boxes:
        tx1, ty1, tx2, ty2 = map(int, tb.xyxy[0])
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        tboxes.append((tx1, ty1, tx2, ty2, float(tb.conf[0])))
        for i, (px1, py1, px2, py2) in enumerate(pboxes):
            if px1 - MARGIN <= tcx <= px2 + MARGIN and py1 - MARGIN <= tcy <= py2 + MARGIN:
                pid = int((px1 + px2) / 2) // 60     # 和抛掷demo同款简易ID
                confirm[pid] = confirm.get(pid, 0) + 1
                if confirm[pid] >= CONFIRM_N and in_zone((px1+px2)/2, py2):
                    smoking_ids.add(i)
                break

    plotted = persons.plot()
    for tx1, ty1, tx2, ty2, cf in tboxes:            # 画抽烟目标(红)
        cv2.rectangle(plotted, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
        cv2.putText(plotted, f"{cf:.2f}", (tx1, max(ty1 - 6, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for i in smoking_ids:                             # 画抽烟的人(蓝框+标记)
        px1, py1, px2, py2 = pboxes[i]
        cv2.rectangle(plotted, (px1, py1), (px2, py2), (255, 100, 0), 3)
        cv2.putText(plotted, "SMOKING", (px1, max(py1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 3)

    # 终端告警:抽烟的人有变化时打印一次,并自动存证据图
    if smoking_ids and smoking_ids != last_smoking:
        t = time.strftime("%H:%M:%S")
        print(f"[{t}] [抽烟告警] 检测到 {len(smoking_ids)} 人抽烟")
        snap_id += 1
        fn = f"smoking_alert_{snap_id}.jpg"
        cv2.imwrite(fn, frame)
        print(f"        证据图已保存: {fn}")
    last_smoking = smoking_ids
    plotted = cv2.polylines(plotted, [ZONE], True, (0, 255, 255), 2)
    cv2.imshow("smoking_demo (s=快照 ESC=退出)", plotted)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        snap_id += 1
        cv2.imwrite(f"snapshot_{snap_id}.jpg", frame)
        print(f"已保存 snapshot_{snap_id}.jpg")
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
