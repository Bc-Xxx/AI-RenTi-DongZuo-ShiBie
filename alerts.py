# -*- coding: utf-8 -*-
"""共享输出(公共基础设施): 最新JPEG帧(MJPEG推流用) + 告警记录与证据截图。
引擎线程写入,Web线程读取,全部经锁保护。"""
import datetime
import os
import threading
import time

import cv2

from config import ALERT_LIMIT, SNAP_DIR
os.makedirs(SNAP_DIR, exist_ok=True)

out_lock = threading.Lock()
LATEST_JPEG = None
LAST_FRAME_TS = 0.0
ALERTS = []                        # [{time,type,text,img}] 最新在前


def publish_jpeg(jpeg):
    """引擎每处理完一帧调用: 更新推流用的最新画面"""
    global LATEST_JPEG, LAST_FRAME_TS
    with out_lock:
        LATEST_JPEG = jpeg.tobytes()
        LAST_FRAME_TS = time.time()


def add_alert(atype, text, raw, marked, cam_id=""):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_f = os.path.join(SNAP_DIR, f"{ts}_{cam_id}_{atype}_raw.jpg")
    mk_f = os.path.join(SNAP_DIR, f"{ts}_{cam_id}_{atype}_marked.jpg")
    cv2.imwrite(raw_f, raw)
    cv2.imwrite(mk_f, marked)
    ALERTS.insert(0, {"time": datetime.datetime.now().strftime("%m-%d %H:%M:%S"),
                      "type": atype, "text": text, "img": os.path.basename(mk_f)})
    del ALERTS[ALERT_LIMIT:]
    print(f">>> [{atype}] {text} 截图: {mk_f}")
