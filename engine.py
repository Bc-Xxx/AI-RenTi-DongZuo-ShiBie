# -*- coding: utf-8 -*-
"""检测引擎主循环(编排层): 取流热切换 → 证据计算 → 各检测模块规则 → 绘制 → 告警。
本层不做具体判定: 推理在evidence,规则在modules/,绘制在viz/。
新增检测模块的接入点见 modules/__init__.py 的说明。"""
import time
from collections import defaultdict

import cv2

import alerts
import config
import evidence
import viz
import zones
from capture import LiveCapture
from config import (ALERT_COOLDOWN, ITEM_CLASSES, SMOKING_COOLDOWN,
                    THROW_PAIR_WINDOW)
from modules import climb, smoke
from modules import throw as throw_mod


def engine():
    det, pose = evidence.load_models()
    cam, cur_sig = None, None          # cur_sig=(相机id,取流地址),变化即热切换
    track_fresh = True
    state = defaultdict(lambda: {"grip": 0, "miss": 0, "prev_feet": None,
                                 "prev_wrist": None, "last_climb": -1e9,
                                 "smoke": 0})
    last_alert = {"climb": 0.0, "climb_warn": 0.0, "throw": 0.0,
                  "throw_warn": 0.0, "smoking": 0.0}
    prev_gray, prev_blobs = None, []   # 过带帧差状态(上一帧灰度图/亮斑)
    last_motion_t = 0.0                # 最近一次挥臂命中时刻(与过带配对用)
    fps, t_prev = 0.0, time.time()

    while True:
        c = config.merged_view()
        sig = (c["active_cam"], c["rtsp"])
        if sig != cur_sig:             # 切换相机/修改取流地址:重连+清跟踪态+重载黄框
            cur_sig = sig
            if cam is not None:
                cam.release()
            cam = LiveCapture(c["rtsp"])
            state.clear()
            track_fresh = True
            prev_gray, prev_blobs = None, []
            zones.load_boxes_for(c["active_cam"])
            print(f">>> 切换到相机[{c['cam_name']}] {c['rtsp'] or '(空地址,等待配置)'}")

        frame = cam.read()
        if frame is None:
            time.sleep(0.1)
            continue
        t_now = time.time()

        # ---------- 证据计算(evidence层,与规则无关) ----------
        smoke_tgts = evidence.smoke_targets(
            frame, c["smoke_target_conf"]) if c["enable_smoke"] else []
        result = det.track(frame, classes=[0] + list(ITEM_CLASSES),
                           persist=not track_fresh, conf=c["det_conf"],
                           imgsz=480, verbose=False)[0]
        track_fresh = False
        ids = result.boxes.id
        labels = []
        climb_warn = cross_alarm = throw_hit = False
        smoking_hits = 0
        prects = evidence.person_rects(result)
        items = evidence.item_rects(result) if c["enable_throw"] else []

        # ---------- 逐人规则(modules层,纯判定) ----------
        if ids is not None:
            for box, tid in zip(result.boxes.xyxy, ids.int().tolist()):
                x1, y1, x2, y2 = map(int, box)
                st = state[tid]
                st["miss"] = 0
                feet = ((x1 + x2) / 2.0, float(y2))
                if y2 - y1 < c["min_person_h"]:
                    st["prev_feet"] = feet
                    continue
                cur_in = zones.in_zone(*feet)
                prev_in = zones.in_zone(*st["prev_feet"]) if st["prev_feet"] else False

                # 姿态级联:攀爬(手腕过肩)与抛物(手腕速度)共用同一次推理,只算一次
                grip_now, wrist_speed, wrist_above = False, 0.0, False
                need_pose = (c["enable_climb"] or c["enable_throw"]) \
                    and (cur_in or prev_in)
                if need_pose:
                    grip_now, wrist_speed, wrist_above, wrs0, found = \
                        evidence.pose_person(frame, (x1, y1, x2, y2), pose,
                                             st["prev_wrist"])
                    if found:
                        st["prev_wrist"] = wrs0
                else:
                    st["prev_wrist"] = None

                if c["enable_climb"]:
                    ca, cw = climb.step(st, (x1, y1, x2, y2), feet,
                                        cur_in, prev_in, grip_now, c, t_now, labels)
                    cross_alarm |= ca
                    climb_warn |= cw
                if c["enable_throw"]:
                    if throw_mod.person_step((x1, y1, x2, y2), cur_in,
                                             wrist_speed, wrist_above,
                                             items, c, labels):
                        throw_hit = True
                if c["enable_smoke"]:
                    if smoke.person_step(st, (x1, y1, x2, y2),
                                         smoke_tgts, c, labels):
                        smoking_hits += 1

                st["prev_feet"] = feet

        for tid in list(state):
            state[tid]["miss"] += 1
            if state[tid]["miss"] > 30:
                del state[tid]
        if throw_hit:
            last_motion_t = t_now

        # 物体过带(抛物佐证,纯函数):仅过栏不告警,与挥臂配对才升级红色告警
        band_hit, band_seg = False, None
        if c["enable_band"]:
            band_hit, band_seg, prev_blobs, prev_gray = throw_mod.band_cross(
                frame, prects, prev_gray, prev_blobs, zones.BOXES,
                c["band_min_area"], c["band_speed"])
        throw_paired = bool(band_hit and throw_hit
                            and (t_now - last_motion_t) < THROW_PAIR_WINDOW)
        if throw_hit and not throw_paired:    # 无物体佐证:画面标签降为预警色
            labels = [(a, b2, t.replace("THROW!", "THROW?"), (0, 200, 255))
                      if t.startswith("THROW!") else (a, b2, t, col)
                      for a, b2, t, col in labels]

        # ---------- 绘制与推流 ----------
        plotted = viz.draw_overlay(result.plot(), c, zones.BOXES, smoke_tgts,
                                   labels, band_seg, fps)
        ok, jpeg = cv2.imencode(".jpg", plotted, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            alerts.publish_jpeg(jpeg)

        # ---------- 告警(带5秒冷却,防连刷;文本/截图名带相机标识) ----------
        cid = c["active_cam"]
        if cross_alarm and t_now - last_alert["climb"] > ALERT_COOLDOWN:
            alerts.add_alert("climb", f"[{c['cam_name']}] 有人翻越围栏！",
                             frame, plotted, cid)
            last_alert["climb"] = t_now
        elif climb_warn and t_now - last_alert["climb_warn"] > ALERT_COOLDOWN:
            alerts.add_alert("climb_warn", f"[{c['cam_name']}] 围栏边出现攀爬姿态",
                             frame, plotted, cid)
            last_alert["climb_warn"] = t_now
        if throw_paired and t_now - last_alert["throw"] > ALERT_COOLDOWN:
            alerts.add_alert("throw", f"[{c['cam_name']}] 抛掷动作 + 物体飞越围栏",
                             frame, plotted, cid)
            last_alert["throw"] = t_now
        elif throw_hit and t_now - last_alert["throw_warn"] > 5 \
                and t_now - last_alert["throw"] > 5:
            alerts.add_alert("throw_warn",
                             f"[{c['cam_name']}] 疑似抛掷动作(未捕捉到物体过栏)",
                             frame, plotted, cid)
            last_alert["throw_warn"] = t_now
        if smoking_hits and t_now - last_alert["smoking"] > SMOKING_COOLDOWN:
            alerts.add_alert("smoking", f"[{c['cam_name']}] 抽烟违规: {smoking_hits} 人",
                             frame, plotted, cid)
            last_alert["smoking"] = t_now

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
