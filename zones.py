# -*- coding: utf-8 -*-
"""黄框布防区(公共基础设施): 每个相机独立的标定存储 + 几何判定工具。
存于climb_config.json的{"cams":{相机id:[框...]}},旧单相机格式自动迁移。
检测规则(攀爬/过带)只调用这里的几何函数,不自己存标定。"""
import json
import os

import cv2
import numpy as np

import config

BOXES = []              # 当前相机的布防区,切换相机时整体重建


def make_box(pts):
    arr = np.array(pts, np.int32)
    best_i, best_y = 0, 1e18
    for i in range(len(pts)):                 # y均值最小的边=这个框最上边(墙头触发线)
        a, b = pts[i], pts[(i + 1) % len(pts)]
        if (a[1] + b[1]) / 2.0 < best_y:
            best_y, best_i = (a[1] + b[1]) / 2.0, i
    return {"pts": pts, "arr": arr,
            "top": (pts[best_i], pts[(best_i + 1) % len(pts)])}


def climb_data():
    if os.path.exists(config.CLIMB_CFG):
        try:
            return json.load(open(config.CLIMB_CFG, encoding="utf-8"))
        except Exception as e:
            print(f"黄框标定读取失败: {e}")
    return {}


def load_boxes_for(cam_id):
    """把指定相机的黄框载入BOXES(引擎启动/切换相机时调用)"""
    data = climb_data()
    if "cams" in data:
        pts_list = data["cams"].get(cam_id, [])
    else:                                     # 兼容旧单相机格式
        pts_list = data.get("boxes", [])
    BOXES.clear()
    for pts in pts_list:
        if len(pts) >= 3:
            BOXES.append(make_box(pts))


def migrate_climb_cfg():
    """旧格式{boxes:[..]} -> {cams:{相机id:[..]}},旧框归属启动时的当前相机"""
    data = climb_data()
    if data and "cams" not in data:
        data = {"cams": {config.cfg["active_cam"]: data.get("boxes", [])}}
        json.dump(data, open(config.CLIMB_CFG, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"{config.CLIMB_CFG} 已升级为多相机格式")


def save_cam_boxes(cam_id, new_pts_list, append):
    """网页画完黄框:写进指定相机在climb_config.json里的条目并热加载 → 返回框数"""
    data = climb_data()
    if "cams" not in data:
        data = {"cams": {}}
    old = data["cams"].get(cam_id, []) if append else []
    data["cams"][cam_id] = old + [p for p in new_pts_list if len(p) >= 3]
    json.dump(data, open(config.CLIMB_CFG, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    load_boxes_for(cam_id)
    return len(BOXES)


def drop_cam_boxes(used_ids):
    """删除相机后顺手清掉其黄框条目"""
    data = climb_data()
    if "cams" in data and set(data["cams"]) - set(used_ids):
        data["cams"] = {k: b for k, b in data["cams"].items() if k in used_ids}
        json.dump(data, open(config.CLIMB_CFG, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)


migrate_climb_cfg()
load_boxes_for(config.cfg["active_cam"])
print(f"已加载{len(BOXES)}个黄框(相机[{config.active_cam_obj()['name']}],来自{config.CLIMB_CFG})")


# ---------------- 几何判定工具 ----------------
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
