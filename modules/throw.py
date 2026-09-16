# -*- coding: utf-8 -*-
"""抛物检测规则(纯规则,可单测),两部分:

1. person_step 单人挥臂判定: 手腕速度尖峰 + 手腕高于肩 = 快速挥臂。
   挥臂动作是抛物告警的必要条件——没有动作就没有抛物告警。
2. band_cross 物体过带(帧差,佐证): 全画面帧差找运动亮斑(先抠掉人区),
   前后帧亮斑位移线段跨过墙头触发线 = 物体飞越围栏。
   仅过带不告警(影子/脚步等干扰太多),只在画面留轨迹线;与挥臂配对才升级红色告警。

两级判定(引擎负责组装):
  仅挥臂              → 橙色预警 throw_warn
  挥臂 + 3秒内物体过栏 → 红色告警 throw
"""
import math

import cv2
import numpy as np

from config import (BAND_DIFF_TH, BAND_MASK_PAD, BAND_MAX_BLOB_AREA,
                    BAND_MAX_PAIR_DIST, BAND_SCENE_CUT_MEAN)
from zones import seg_intersect

_K3 = np.ones((3, 3), np.uint8)   # 帧差形态学开运算核(滤单像素噪点,实现细节不属配置)


def person_step(box, cur_in, wrist_speed, wrist_above, items, c, labels):
    """单人挥臂判定 → throw_hit(bool)。
    items: 落在画面里的背包/手提包/箱类框(持物是辅助标注,不阻断告警)"""
    x1, y1, _, _ = box
    if not cur_in:                        # 只看布防区内的人
        return False
    held = any(ix1 > x1 and ix2 < x2 and iy1 > y1 and iy2 < y2
               for ix1, iy1, ix2, iy2 in items)
    if wrist_speed > c["spike"] and wrist_above:
        labels.append((x1, y1, f"THROW!{'(持物)' if held else ''}", (0, 0, 255)))
        return True
    if held:
        labels.append((x1, y1, "持物", (0, 180, 255)))
    return False


def band_cross(frame, person_rects, prev_gray, prev_blobs, boxes, min_area, speed):
    """物体过带(纯函数,可单测) → (band_hit, band_seg, cur_blobs, prev_gray)

    frame: 当前帧BGR; person_rects: 全部人框(帧差里抠掉,人体动作不算物体);
    prev_gray/prev_blobs: 上一帧灰度图与亮斑(引擎跨帧保存,切相机时重置);
    boxes: zones.BOXES(墙头触发线在其中); min_area/speed: band_*参数"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not boxes or prev_gray is None:
        return False, None, [], gray
    diff = cv2.absdiff(prev_gray, gray)
    if diff.mean() > BAND_SCENE_CUT_MEAN:      # 场景突变,本帧不配对
        return False, None, [], gray

    for px1, py1, px2, py2 in person_rects:
        cv2.rectangle(diff, (px1 - BAND_MASK_PAD, py1 - BAND_MASK_PAD),
                      (px2 + BAND_MASK_PAD, py2 + BAND_MASK_PAD), 0, -1)
    bw = cv2.morphologyEx(
        cv2.threshold(diff, BAND_DIFF_TH, 255, cv2.THRESH_BINARY)[1],
        cv2.MORPH_OPEN, _K3)
    nlab, _, stats, cent = cv2.connectedComponentsWithStats(bw)
    cur = []
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= BAND_MAX_BLOB_AREA:
            cur.append((cent[i][0], cent[i][1],
                        int(stats[i, cv2.CC_STAT_WIDTH]),
                        int(stats[i, cv2.CC_STAT_HEIGHT])))

    for pcx, pcy, *_ in prev_blobs:       # 前后帧亮斑全配对
        for ccx, ccy, cw, ch in cur:
            dx, dy = ccx - pcx, ccy - pcy
            dist = math.hypot(dx, dy)
            if not speed <= dist <= BAND_MAX_PAIR_DIST:
                continue
            if dy > 0 and abs(dx) < 0.3 * abs(dy) and ch > 2.5 * cw:
                continue                   # 竖直向下细长条=雨雪,排除
            for b in boxes:
                if seg_intersect((pcx, pcy), (ccx, ccy),
                                 b["top"][0], b["top"][1]):
                    return True, ((pcx, pcy), (ccx, ccy)), cur, gray
    return False, None, cur, gray
