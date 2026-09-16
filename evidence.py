# -*- coding: utf-8 -*-
"""模型推理层(公共基础设施): YOLO检测/姿态/抽烟模型的加载与逐帧证据计算。
规则模块(modules/)只吃这里输出的证据,不直接碰模型;
未来换模型/加新模型权重,只需要改这一层,规则层不动。"""
import math

from ultralytics import YOLO

from config import (ITEM_CLASSES, L_SH, POSE_MODEL, R_SH, R_WR, SMOKE_MODEL_FILE,
                    DET_MODEL, L_WR)

_smoke_model = None                     # 懒加载:开启模块才加载,缺文件自动跳过


def load_models():
    """主检测模型: 人/物品检测 + 姿态(每进程加载一次,引擎启动时调用)"""
    return YOLO(DET_MODEL), YOLO(POSE_MODEL)


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


def person_rects(result):
    """全部人框(含远处小框) —— 过带帧差抠人、逐人规则遍历都用它"""
    return [tuple(map(int, b)) for b, cl in
            zip(result.boxes.xyxy, result.boxes.cls.int().tolist()) if cl == 0]


def item_rects(result):
    """背包/手提包/行李箱框(抛物模块的持物判断用)"""
    return [tuple(map(int, b.xyxy[0])) for b, cls in
            zip(result.boxes, result.boxes.cls.int().tolist())
            if cls in ITEM_CLASSES]


def smoke_targets(frame, conf):
    """烟头/吸烟目标检出 → [(中心x,中心y,tx1,ty1,tx2,ty2,置信度), ...]"""
    sm = get_smoke_model()
    if sm is None:
        return []
    tr = sm(frame, conf=conf, imgsz=640, verbose=False)[0]
    out = []
    for tb in tr.boxes:
        tx1, ty1, tx2, ty2 = map(int, tb.xyxy[0])
        out.append(((tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0,
                    tx1, ty1, tx2, ty2, float(tb.conf[0])))
    return out


def pose_person(frame, box, pose, prev_wrist):
    """人框小图姿态推理(级联:只对布防区内的人算,省算力)
    → (grip_now手腕过肩, wrist_speed像素/帧, wrist_above, 本帧手腕位置, 是否有效)
    关键点是crop坐标系,这里已换算回整帧坐标"""
    x1, y1, x2, y2 = box
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return False, 0.0, False, None, False
    pr = pose(crop, imgsz=320, verbose=False)[0]
    kpts = pr.keypoints
    if not (kpts is not None and kpts.data is not None
            and kpts.data.numel() > 0
            and kpts.xy.shape[0] > 0 and kpts.xy.shape[1] >= 17):
        return False, 0.0, False, None, False
    kpt = kpts.xy[0]
    sh_ys = [float(kpt[i][1]) + y1 for i in (L_SH, R_SH) if kpt[i][1] > 0]
    wrs = [(float(kpt[i][0]) + x1, float(kpt[i][1]) + y1)
           for i in (L_WR, R_WR) if kpt[i][0] > 0]
    if not (sh_ys and wrs):
        return False, 0.0, False, None, False
    wrist_above = any(w[1] < min(sh_ys) for w in wrs)
    wrist_speed = max(math.dist(w, prev_wrist) for w in wrs) if prev_wrist else 0.0
    return wrist_above, wrist_speed, wrist_above, wrs[0], True
