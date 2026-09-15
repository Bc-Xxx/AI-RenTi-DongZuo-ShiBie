# -*- coding: utf-8 -*-
"""
人员攀爬/翻越检测演示(多黄框版,支持拐角/多段围栏)

用法:按 z 画一个黄框圈住一段围栏(梯子) —— 每个框自己的"最上边"自动成为
     墙头线(红色加粗显示)。拐角/两面墙 = 画两个框,每面墙各有一条触发线。
判定:
  预警级: 人进入任一黄框 + 双手腕持续高于肩线 >= GRIP_FRAMES 帧
  告警级: 黄框内的人脚底轨迹穿过所在框的最上边 + POSTURE_WINDOW 秒内
          出现过攀爬姿态(跨坐墙头手臂垂下也有效)
操作键(在视频窗口上按,需切英文输入法):
  z   画/追加一个黄框 —— 左键逐点,回车完成(>=3点),u撤销,ESC取消
  x   撤销最后一个黄框
  c   清空全部黄框
  ESC 退出程序
标定自动保存到 climb_config.json,重启自动加载
"""
import os
import time
import json
import datetime
import threading
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"  # 强制TCP防丢包

RTSP = "rtsp://<用户名>:<密码>@<相机IP>:554/Streaming/Channels/301"
CFG_FILE = "climb_config.json"
SNAPSHOT_DIR = "climb_shots"        # 告警截图保存目录
MIN_SNAP_INTERVAL = 5.0             # 两次截图最小间隔(秒)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def save_snapshot(raw, marked, tag):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"{ts}_{tag}_raw.jpg"), raw)
    cv2.imwrite(os.path.join(SNAPSHOT_DIR, f"{ts}_{tag}_marked.jpg"), marked)
    print(f"已保存截图 -> {SNAPSHOT_DIR}/{ts}_{tag}_raw.jpg / _marked.jpg")

# ---------------- 配置 ----------------
CFG = {"boxes": [], "grip_frames": 6, "require_posture": 1, "posture_window": 6}
if os.path.exists(CFG_FILE):
    try:
        CFG.update(json.load(open(CFG_FILE, encoding="utf-8")))
        print(f"已加载配置 {CFG_FILE}")
    except Exception as e:
        print(f"配置文件读取失败,使用默认: {e}")

def save_cfg():
    json.dump(CFG, open(CFG_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"已保存配置 -> {CFG_FILE}")

GRIP_FRAMES = CFG["grip_frames"]
REQUIRE_POSTURE = bool(CFG.get("require_posture", 1))   # 告警是否要求姿态证据(夜间可设0)
POSTURE_WINDOW = float(CFG.get("posture_window", 6))    # 过线前N秒内出现过攀爬姿态即有效
DET_CONF = 0.45          # 人检测置信度阈值(默认0.25太宽松;误框多就调大,漏检远处的人就调小)
MIN_PERSON_H = 50        # 人框最小高度(像素),过滤远处小误框;按现场相机标定
L_SH, R_SH = 5, 6       # 左右肩
L_WR, R_WR = 9, 10      # 左右手腕

# ---------------- 黄框(可多个) ----------------
BOXES = []              # 每项: {"pts":顶点, "arr":数组, "top":(墙头线端点1, 端点2)}

def make_box(pts):
    arr = np.array(pts, np.int32)
    best_i, best_y = 0, 1e18
    for i in range(len(pts)):                 # y均值最小的边=这个框的最上边
        a, b = pts[i], pts[(i + 1) % len(pts)]
        ay = (a[1] + b[1]) / 2.0
        if ay < best_y:
            best_y, best_i = ay, i
    return {"pts": pts, "arr": arr, "top": (pts[best_i], pts[(best_i + 1) % len(pts)])}

def rebuild_boxes():
    BOXES.clear()
    for pts in CFG["boxes"]:
        if len(pts) >= 3:
            BOXES.append(make_box(pts))

rebuild_boxes()

# ---------------- 几何工具 ----------------
def in_zone(x, y):
    """点是否在任一黄框内"""
    for b in BOXES:
        if cv2.pointPolygonTest(b["arr"], (float(x), float(y)), False) >= 0:
            return True
    return False

def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

def seg_intersect(p1, p2, p3, p4):
    d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
    d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

def cross_any_top(prev_pt, cur_pt):
    """脚底轨迹是否穿过任一黄框的墙头线"""
    if prev_pt is None:
        return False
    for b in BOXES:
        if seg_intersect(prev_pt, cur_pt, b["top"][0], b["top"][1]):
            return True
    return False

# ---------------- 低延迟取流 ----------------
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
                print("画面中断，尝试重连...")
                self.cap.release()
                time.sleep(0.5)
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


det = YOLO("yolo11s.pt")
pose = YOLO("yolo11s-pose.pt")

DRAW = {"mode": None, "pts": []}

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and DRAW["mode"]:
        DRAW["pts"].append([x, y])

state = defaultdict(lambda: {"grip": 0, "miss": 0, "prev_feet": None, "last_climb": -1e9})
cooldown = 0
prev_cross = prev_warn = False
last_snap = 0.0
fps, t_prev = 0.0, time.time()

print(f"正在连接相机: {RTSP}")
cam = LiveCapture(RTSP)

cv2.namedWindow("climb_demo")
cv2.setMouseCallback("climb_demo", on_mouse)

while True:
    frame = cam.read()
    if frame is None:
        cv2.waitKey(100)
        continue
    H, W = frame.shape[:2]

    # ---------- 画图模式 ----------
    if DRAW["mode"]:
        show = frame.copy()
        pts = DRAW["pts"]
        # 已有框画淡色,正在画的画亮色
        for b in BOXES:
            cv2.polylines(show, [b["arr"]], True, (0, 160, 160), 1)
            cv2.line(show, tuple(map(int, b["top"][0])), tuple(map(int, b["top"][1])),
                     (0, 0, 200), 2)
        if len(pts) >= 2:
            cv2.polylines(show, [np.array(pts, np.int32)], True, (0, 255, 255), 2)
        for p in pts:
            cv2.circle(show, tuple(p), 4, (0, 255, 255), -1)
        cv2.putText(show, "黄框:左键加点(圈住一段围栏,顶边压在墙头) 回车完成 u撤销 ESC取消",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("climb_demo", show)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:                                     # 回车=完成,追加一个框
            if len(pts) >= 3:
                CFG["boxes"].append(pts)
                rebuild_boxes()
                save_cfg()
                print(f"已添加第{len(BOXES)}个黄框,继续按 z 可画下一面墙")
            DRAW = {"mode": None, "pts": []}
        elif key == ord('u') and pts:
            pts.pop()
        elif key == 27:
            DRAW = {"mode": None, "pts": []}
        continue

    # ---------- 正常检测 ----------
    t_now = time.time()
    result = det.track(frame, classes=[0], persist=True, conf=DET_CONF,
                       imgsz=480, verbose=False)[0]
    ids = result.boxes.id
    labels = []
    climb_warn = cross_alarm = False

    if ids is not None:
        for box, tid in zip(result.boxes.xyxy, ids.int().tolist()):
            x1, y1, x2, y2 = map(int, box)
            if y2 - y1 < MIN_PERSON_H:                # 太小=远处误框,跳过
                continue
            st = state[tid]
            st["miss"] = 0
            feet = ((x1 + x2) / 2.0, float(y2))           # 脚底中心
            cur_in = in_zone(feet[0], feet[1])
            prev_in = in_zone(*st["prev_feet"]) if st["prev_feet"] else False

            # ---- 姿态级联:框内(或刚在框内)的人跑姿态模型 ----
            grip_now = False
            if cur_in or prev_in:
                crop = frame[max(y1, 0):y2, max(x1, 0):x2]
                if crop.size > 0:
                    pr = pose(crop, imgsz=320, verbose=False)[0]
                    kpts = pr.keypoints
                    if (kpts is not None and kpts.data is not None
                            and kpts.data.numel() > 0
                            and kpts.xy.shape[0] > 0
                            and kpts.xy.shape[1] >= 17):
                        kpt = kpts.xy[0]
                        sh_ys = [kpt[i][1] for i in (L_SH, R_SH) if kpt[i][1] > 0]
                        wrs = [kpt[i] for i in (L_WR, R_WR) if kpt[i][0] > 0]
                        if sh_ys and wrs and any(w[1] < min(sh_ys) for w in wrs):
                            grip_now = True
            st["grip"] = st["grip"] + 1 if grip_now else 0
            if st["grip"] >= GRIP_FRAMES:
                st["last_climb"] = t_now                  # 记住最近一次攀爬姿态的时刻
            # 跨坐墙头时手臂会垂下,姿态证据与过线不同时发生:
            # 改为"过线前 POSTURE_WINDOW 秒内出现过持续攀爬姿态"即有效
            climbing_recent = (t_now - st["last_climb"]) < POSTURE_WINDOW

            # ---- 告警级:脚底穿墙头线 + 最近出现过攀爬姿态(时间窗口证据) ----
            posture_ok = climbing_recent or not REQUIRE_POSTURE
            if cross_any_top(st["prev_feet"], feet) \
                    and (prev_in or cur_in) and posture_ok:
                labels.append((x1, y1, "CLIMB!", (0, 0, 255)))
                cross_alarm = True

            # ---- 预警级:框内当前持续攀爬姿态 ----
            if not cross_alarm and cur_in and st["grip"] >= GRIP_FRAMES:
                labels.append((x1, y1, "CLIMBING", (0, 200, 255)))
                climb_warn = True

            st["prev_feet"] = feet

    for tid in list(state):
        state[tid]["miss"] += 1
        if state[tid]["miss"] > 30:
            del state[tid]

    cooldown -= 1
    if (climb_warn or cross_alarm) and cooldown <= 0:
        if cross_alarm:
            print(">>> [告警] 有人翻越围栏！")
        elif climb_warn:
            print(">>> [预警] 围栏边出现攀爬姿态")
        cooldown = 100

    # ---------- 叠加显示 ----------
    plotted = result.plot()
    for b in BOXES:
        plotted = cv2.polylines(plotted, [b["arr"]], True, (0, 255, 255), 2)
        plotted = cv2.line(plotted, tuple(map(int, b["top"][0])),
                           tuple(map(int, b["top"][1])), (0, 0, 255), 3)
    if not BOXES:
        cv2.putText(plotted, "Press z: draw fence box(es)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for sx, sy, txt, color in labels:
        cv2.putText(plotted, txt, (sx, max(sy - 8, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if cross_alarm:
        cv2.putText(plotted, "FENCE CLIMB!", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(plotted, f"FPS:{fps:.1f}", (W - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.putText(plotted, "z=add box  x=undo box  c=clear  ESC=quit", (20, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 状态出现瞬间截图
    now_snap = time.time()
    if ((cross_alarm and not prev_cross) or (climb_warn and not prev_warn)) \
            and now_snap - last_snap > MIN_SNAP_INTERVAL:
        save_snapshot(frame, plotted, "climb" if cross_alarm else "warn")
        last_snap = now_snap
    prev_cross, prev_warn = cross_alarm, climb_warn

    cv2.imshow("climb_demo", plotted)

    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
    t_prev = now

    key = cv2.waitKey(1) & 0xFF
    if key == ord('z'):
        DRAW = {"mode": "zone", "pts": []}
    elif key == ord('x') and CFG["boxes"]:             # 撤销最后一个框
        CFG["boxes"].pop()
        rebuild_boxes()
        save_cfg()
        print(f"已撤销,剩余{len(BOXES)}个黄框")
    elif key == ord('c'):                              # 清空全部
        CFG["boxes"] = []
        rebuild_boxes()
        save_cfg()
        print("已清空全部黄框,按 z 重新绘制")
    elif key == 27:
        break

cam.release()
cv2.destroyAllWindows()
