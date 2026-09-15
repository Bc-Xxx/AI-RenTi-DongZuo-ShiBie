# -*- coding: utf-8 -*-
"""
安防检测平台 一期：单相机 + 攀爬/抛物模块 + Web界面
运行:
  pip install fastapi uvicorn
  python app.py
浏览器打开 http://localhost:8000
"""
import os
import time
import json
import math
import datetime
import threading
from collections import defaultdict

import cv2
import numpy as np
import uvicorn
from ultralytics import YOLO
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ---------------- 配置 ----------------
CFG_FILE = "web_config.json"
CLIMB_CFG = "climb_config.json"     # 复用攀爬脚本画的黄框标定
SNAP_DIR = "alerts"
os.makedirs(SNAP_DIR, exist_ok=True)

DEFAULT_CFG = {
    "_说明": {
        "总览": "所有参数在网页上拖动滑条会自动改写此文件；也可手动改,改完重启app.py生效。此文件不支持//注释,说明都写在_说明里,不要删",
        "rtsp": "相机取流地址。换相机时改这里(或直接改文件后重启)",
        "enable_climb": "攀爬检测模块开关。true开启/false关闭,网页页签也可切",
        "enable_throw": "抛物检测模块开关。true开启/false关闭",
        "det_conf": "检测置信度阈值(0.25~0.7)。模型对每个目标打分,低于此值的框丢弃。误框多(把杂物框成人)→调大;远处的人/夜间的人漏检→调小。白天0.45左右,夜间可降到0.35",
        "min_person_h": "人框最小高度(像素)。比这矮的检测框直接忽略(大多是远处小误检)。设定原则:比'需要监控的最远真人在画面里的高度'小,比'常见误检杂物的高度'大。换分辨率/换机位后必须重标",
        "spike": "抛物-手腕速度阈值(像素/帧)。相邻两帧手腕移动距离超过此值=快速挥臂。正常挥手约20~40,真抛掷上百。调法:让人做两种动作看告警,卡在中间。帧率变了要重标",
        "grip_frames": "攀爬-姿态持续帧数。手腕高于肩线连续这么多帧才认定攀爬姿态(单帧举手不算)。误报多→调大;反应慢→调小。6帧约0.5~1秒",
        "posture_window": "攀爬-姿态证据有效窗口(秒)。人跨坐墙头时手臂会垂下,过线发生在垂手之后;过线前N秒内出现过攀爬姿态即算证据成立。骑墙磨蹭久会漏→调大;爬完又正常走动的人被误判→调小",
        "require_posture": "攀爬-告警是否要求姿态证据。1=要求(白天推荐,误报少);0=不要求,脚底过墙头线就告警(夜间红外画面姿态不准时的兜底,宁误报不漏报)",
        "enable_smoke": "抽烟检测模块开关。需smoke_best.pt权重文件和app.py同目录,文件缺失时模块自动跳过",
        "smoke_target_conf": "抽烟-目标置信度阈值(0.1~0.6)。烟头/香烟目标小且模糊,过高会漏检,过低误报多;误报靠smoke_confirm_n连续确认压制",
        "smoke_confirm_n": "抽烟-连续确认帧数。抽烟目标连续N帧关联到同一个人身上才告警。误报多→调大;检出慢→调小",
        "show_boxes": "画面上是否显示黄框/墙头线叠加。true显示/false隐藏。只影响显示,隐藏后判定照常工作"
    },
    "rtsp": "rtsp://<用户名>:<密码>@<相机IP>:554/Streaming/Channels/301",
    "enable_climb": True,
    "enable_throw": False,
    "enable_smoke": False,
    "det_conf": 0.45,
    "min_person_h": 50,
    "spike": 40,
    "grip_frames": 6,
    "posture_window": 6,
    "require_posture": 1,
    "smoke_target_conf": 0.30,
    "smoke_confirm_n": 8,
    "show_boxes": True,
}

cfg = dict(DEFAULT_CFG)
if os.path.exists(CFG_FILE):
    try:
        cfg.update(json.load(open(CFG_FILE, encoding="utf-8")))
    except Exception as e:
        print(f"配置读取失败用默认: {e}")

cfg_lock = threading.Lock()

def save_cfg():
    with cfg_lock:
        json.dump(cfg, open(CFG_FILE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

# ---------------- 黄框标定(沿用攀爬脚本画的) ----------------
BOXES = []
if os.path.exists(CLIMB_CFG):
    try:
        for pts in json.load(open(CLIMB_CFG, encoding="utf-8")).get("boxes", []):
            if len(pts) >= 3:
                arr = np.array(pts, np.int32)
                best_i, best_y = 0, 1e18
                for i in range(len(pts)):
                    a, b = pts[i], pts[(i + 1) % len(pts)]
                    if (a[1] + b[1]) / 2.0 < best_y:
                        best_y, best_i = (a[1] + b[1]) / 2.0, i
                BOXES.append({"arr": arr,
                              "top": (pts[best_i], pts[(best_i + 1) % len(pts)])})
        print(f"已加载{len(BOXES)}个黄框(来自{CLIMB_CFG})")
    except Exception as e:
        print(f"黄框标定读取失败: {e}")

def in_zone(x, y):
    if not BOXES:
        return True                # 没画框时视为处处布防
    for b in BOXES:
        if cv2.pointPolygonTest(b["arr"], (float(x), float(y)), False) >= 0:
            return True
    return False

def _cross(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def seg_intersect(p1, p2, p3, p4):
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

def cross_any_top(prev_pt, cur_pt):
    if prev_pt is None or not BOXES:
        return False
    for b in BOXES:
        if seg_intersect(prev_pt, cur_pt, b["top"][0], b["top"][1]):
            return True
    return False

# ---------------- 取流 ----------------
class LiveCapture:
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
                self.cap.release()
                time.sleep(0.5)
                with cfg_lock:
                    self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return self.frame

    def release(self):
        self.running = False
        time.sleep(0.2)
        self.cap.release()

# ---------------- 共享输出 ----------------
out_lock = threading.Lock()
LATEST_JPEG = None
LAST_FRAME_TS = 0.0
ALERTS = []                        # [{time,type,text,img}] 最新在前
ALERT_LIMIT = 100

def add_alert(atype, text, raw, marked):
    global LAST_ALERT_TS
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_f = os.path.join(SNAP_DIR, f"{ts}_{atype}_raw.jpg")
    mk_f = os.path.join(SNAP_DIR, f"{ts}_{atype}_marked.jpg")
    cv2.imwrite(raw_f, raw)
    cv2.imwrite(mk_f, marked)
    ALERTS.insert(0, {"time": datetime.datetime.now().strftime("%m-%d %H:%M:%S"),
                      "type": atype, "text": text, "img": os.path.basename(mk_f)})
    del ALERTS[ALERT_LIMIT:]
    print(f">>> [{atype}] {text} 截图: {mk_f}")

# ---------------- 检测引擎 ----------------
L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10
ITEM_CLASSES = {24: "背包", 26: "手提包", 28: "箱"}   # COCO: backpack/handbag/suitcase
SMOKE_MODEL_FILE = "smoke_best.pt"   # 抽烟检测权重(需和app.py同目录)
SMOKE_MARGIN = 30                    # 抽烟目标允许超出人框的像素余量
_smoke_model = None                  # 懒加载:开启模块才加载,缺文件自动跳过

def get_smoke_model():
    global _smoke_model
    if _smoke_model is None:
        try:
            _smoke_model = YOLO(SMOKE_MODEL_FILE)
            print("抽烟模型已加载,类别:", _smoke_model.names)
        except Exception as e:
            print(f"抽烟模型加载失败(模块将跳过): {e}")
            _smoke_model = False     # 标记失败,避免每帧重试
    return _smoke_model if _smoke_model else None

def engine():
    global LATEST_JPEG, LAST_FRAME_TS
    det = YOLO("yolo11s.pt")
    pose = YOLO("yolo11s-pose.pt")
    with cfg_lock:
        cam = LiveCapture(cfg["rtsp"])
    state = defaultdict(lambda: {"grip": 0, "miss": 0, "prev_feet": None,
                                 "prev_wrist": None, "last_climb": -1e9,
                                 "smoke": 0})
    last_alert = {"climb": 0.0, "climb_warn": 0.0, "throw": 0.0, "smoking": 0.0}
    fps, t_prev = 0.0, time.time()

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.1)
            continue
        H, W = frame.shape[:2]
        with cfg_lock:
            c = dict(cfg)
        t_now = time.time()

        # ---- 抽烟模块:烟头/吸烟目标检测(开启才推理,缺权重自动跳过) ----
        smoke_targets = []
        if c["enable_smoke"]:
            sm = get_smoke_model()
            if sm is not None:
                tr = sm(frame, conf=c["smoke_target_conf"], imgsz=640,
                        verbose=False)[0]
                for tb in tr.boxes:
                    tx1, ty1, tx2, ty2 = map(int, tb.xyxy[0])
                    smoke_targets.append(((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0,
                                          tx1, ty1, tx2, ty2, float(tb.conf[0])))

        result = det.track(frame, classes=[0] + list(ITEM_CLASSES),
                           persist=True, conf=c["det_conf"],
                           imgsz=480, verbose=False)[0]
        ids = result.boxes.id
        labels = []
        events = []                 # (atype, text) 本帧产生的告警
        climb_warn = cross_alarm = throw_hit = False
        smoking_hits = 0

        if ids is not None:
            # 物品框先收集(用于抛物模块的持物判断)
            items = [tuple(map(int, b.xyxy[0])) for b, cls in
                     zip(result.boxes, result.boxes.cls.int().tolist())
                     if cls in ITEM_CLASSES] if c["enable_throw"] else []

            for box, tid in zip(result.boxes.xyxy, ids.int().tolist()):
                x1, y1, x2, y2 = map(int, box)
                st = state[tid]
                st["miss"] = 0
                feet = ((x1 + x2) / 2.0, float(y2))
                if y2 - y1 < c["min_person_h"]:
                    st["prev_feet"] = feet
                    continue
                cur_in = in_zone(*feet)
                prev_in = in_zone(*st["prev_feet"]) if st["prev_feet"] else False

                # ---- 姿态级联:两个模块都需要,只算一次 ----
                grip_now = False
                wrist_speed = 0.0
                wrist_above = False
                need_pose = (c["enable_climb"] or c["enable_throw"]) and (cur_in or prev_in)
                if need_pose:
                    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                    if crop.size > 0:
                        pr = pose(crop, imgsz=320, verbose=False)[0]
                        kpts = pr.keypoints
                        if (kpts is not None and kpts.data is not None
                                and kpts.data.numel() > 0
                                and kpts.xy.shape[0] > 0 and kpts.xy.shape[1] >= 17):
                            kpt = kpts.xy[0]
                            # 关键点是crop坐标系,换算回整帧坐标
                            sh_ys = [float(kpt[i][1]) + y1 for i in (L_SH, R_SH)
                                     if kpt[i][1] > 0]
                            wrs = [(float(kpt[i][0]) + x1, float(kpt[i][1]) + y1)
                                   for i in (L_WR, R_WR) if kpt[i][0] > 0]
                            if sh_ys and wrs:
                                sh_min = min(sh_ys)
                                wrist_above = any(w[1] < sh_min for w in wrs)
                                grip_now = wrist_above
                                if st["prev_wrist"]:
                                    wrist_speed = max(math.dist(w, st["prev_wrist"])
                                                      for w in wrs)
                                st["prev_wrist"] = wrs[0]
                else:
                    st["prev_wrist"] = None

                # ---- 攀爬模块 ----
                if c["enable_climb"]:
                    st["grip"] = st["grip"] + 1 if grip_now else 0
                    if st["grip"] >= c["grip_frames"]:
                        st["last_climb"] = t_now
                    climbing_recent = (t_now - st["last_climb"]) < c["posture_window"]
                    posture_ok = climbing_recent or not c["require_posture"]
                    if cross_any_top(st["prev_feet"], feet) and (prev_in or cur_in) \
                            and posture_ok:
                        labels.append((x1, y1, "CLIMB!", (0, 0, 255)))
                        cross_alarm = True
                    elif cur_in and st["grip"] >= c["grip_frames"]:
                        labels.append((x1, y1, "CLIMBING", (0, 200, 255)))
                        climb_warn = True

                # ---- 抛物模块(一期:手腕速度尖峰+持物关联) ----
                if c["enable_throw"] and cur_in:
                    held = any(ix1 > x1 and ix2 < x2 and iy1 > y1 and iy2 < y2
                               for ix1, iy1, ix2, iy2 in items)
                    if wrist_speed > c["spike"] and wrist_above:
                        labels.append((x1, y1, f"THROW!{'(持物)' if held else ''}",
                                       (0, 0, 255)))
                        throw_hit = True
                    elif held:
                        labels.append((x1, y1, "持物", (0, 180, 255)))

                # ---- 抽烟模块:目标连续N帧关联到同一个人 ----
                if c["enable_smoke"] and smoke_targets:
                    hit = any(x1 - SMOKE_MARGIN <= tcx <= x2 + SMOKE_MARGIN and
                              y1 - SMOKE_MARGIN <= tcy <= y2 + SMOKE_MARGIN
                              for tcx, tcy, *_ in smoke_targets)
                    st["smoke"] = st["smoke"] + 1 if hit else 0
                    if st["smoke"] >= c["smoke_confirm_n"]:
                        labels.append((x1, y1, "SMOKING", (200, 80, 255)))
                        smoking_hits += 1

                st["prev_feet"] = feet

        for tid in list(state):
            state[tid]["miss"] += 1
            if state[tid]["miss"] > 30:
                del state[tid]

        # ---- 绘制 ----
        plotted = result.plot()
        if c.get("show_boxes", True):          # 标定框显示开关(只管显示)
            for b in BOXES:
                plotted = cv2.polylines(plotted, [b["arr"]], True, (0, 255, 255), 2)
                plotted = cv2.line(plotted, tuple(map(int, b["top"][0])),
                                   tuple(map(int, b["top"][1])), (0, 0, 255), 3)
        for tcx, tcy, tx1, ty1, tx2, ty2, cf in smoke_targets:
            cv2.rectangle(plotted, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
            cv2.putText(plotted, f"{cf:.2f}", (tx1, max(ty1 - 6, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for sx, sy, txt, color in labels:
            cv2.putText(plotted, txt, (sx, max(sy - 8, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        mods = []
        if c["enable_climb"]: mods.append("攀爬")
        if c["enable_throw"]: mods.append("抛物")
        if c["enable_smoke"]: mods.append("抽烟")
        cv2.putText(plotted, f"MOD:{'+'.join(mods) or 'none'} FPS:{fps:.1f}",
                    (W - 420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        if not BOXES:
            cv2.putText(plotted, "no fence box (all zone)", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        ok, jpeg = cv2.imencode(".jpg", plotted, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with out_lock:
                LATEST_JPEG = jpeg.tobytes()
                LAST_FRAME_TS = time.time()

        # ---- 告警(带5秒冷却,防连刷) ----
        if cross_alarm and t_now - last_alert["climb"] > 5:
            add_alert("climb", "有人翻越围栏！", frame, plotted)
            last_alert["climb"] = t_now
        elif climb_warn and t_now - last_alert["climb_warn"] > 5:
            add_alert("climb_warn", "围栏边出现攀爬姿态", frame, plotted)
            last_alert["climb_warn"] = t_now
        elif throw_hit and t_now - last_alert["throw"] > 5:
            add_alert("throw", "布防区内出现抛掷动作", frame, plotted)
            last_alert["throw"] = t_now
        if smoking_hits and t_now - last_alert["smoking"] > 10:
            add_alert("smoking", f"抽烟违规: {smoking_hits} 人", frame, plotted)
            last_alert["smoking"] = t_now

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

# ---------------- Web服务 ----------------
app = FastAPI()

class CfgIn(BaseModel):
    enable_climb: bool | None = None
    enable_throw: bool | None = None
    enable_smoke: bool | None = None
    det_conf: float | None = None
    min_person_h: int | None = None
    spike: float | None = None
    grip_frames: int | None = None
    posture_window: float | None = None
    require_posture: int | None = None
    smoke_target_conf: float | None = None
    smoke_confirm_n: int | None = None
    show_boxes: bool | None = None

@app.get("/")
def index():
    return FileResponse("index.html", media_type="text/html")

@app.get("/api/config")
def get_config():
    with cfg_lock:
        return cfg

@app.post("/api/config")
def set_config(v: CfgIn):
    with cfg_lock:
        for k, val in v.dict().items():
            if val is not None:
                cfg[k] = val
    save_cfg()
    return {"ok": True}

@app.get("/api/alerts")
def get_alerts():
    with out_lock:
        return list(ALERTS)

@app.get("/api/status")
def status():
    with out_lock:
        age = time.time() - LAST_FRAME_TS
    return {"alive": age < 3.0, "fps_age": round(age, 2)}

@app.get("/alerts/{name}")
def alert_img(name: str):
    p = os.path.join(SNAP_DIR, os.path.basename(name))
    if not os.path.exists(p):
        return Response(status_code=404)
    return FileResponse(p, media_type="image/jpeg")

class BoxesIn(BaseModel):
    boxes: list
    append: bool = False       # True=追加到已有框(画拐角第二面墙), False=整体替换

@app.get("/api/boxes")
def get_boxes():
    return {"boxes": [b.get("pts", []) for b in BOXES]}

@app.post("/api/boxes")
def set_boxes(v: BoxesIn):
    """网页画完黄框:写回climb_config.json并热加载"""
    old = []
    if v.append:
        try:
            old = json.load(open(CLIMB_CFG, encoding="utf-8")).get("boxes", [])
        except Exception:
            old = []
    new_boxes = old + [p for p in v.boxes if len(p) >= 3]
    json.dump({"boxes": new_boxes}, open(CLIMB_CFG, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    BOXES.clear()
    for pts in new_boxes:
        arr = np.array(pts, np.int32)
        best_i, best_y = 0, 1e18
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            if (a[1] + b[1]) / 2.0 < best_y:
                best_y, best_i = (a[1] + b[1]) / 2.0, i
        BOXES.append({"pts": pts, "arr": arr,
                      "top": (pts[best_i], pts[(best_i + 1) % len(pts)])})
    return {"ok": True, "count": len(BOXES)}

def mjpeg():
    while True:
        with out_lock:
            j = LATEST_JPEG
        if j:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
        time.sleep(0.05)

@app.get("/api/stream")
def stream():
    return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

threading.Thread(target=engine, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
