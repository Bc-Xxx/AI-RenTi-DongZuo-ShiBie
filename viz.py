# -*- coding: utf-8 -*-
"""画面叠加绘制(公共基础设施): 中文角标/标定框/模块行/告警标签/过带轨迹。
cv2.putText画不了中文,中文走PIL。"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT = None


def put_text_cn(img, text, x, y, color=(255, 210, 120), size=24):
    global _FONT
    if _FONT is None:
        for f in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
            try:
                _FONT = ImageFont.truetype(f, size)
                break
            except Exception:
                continue
        if _FONT is None:
            _FONT = ImageFont.load_default()
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((x, y), text, font=_FONT,
                             fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_overlay(plotted, c, boxes, smoke_targets, labels, band_seg, fps):
    """把公共叠加元素画到推理结果图上 → 返回画好的图"""
    W = plotted.shape[1]
    if c.get("show_boxes", True):          # 标定框显示开关(只管显示)
        for b in boxes:
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
    if band_seg:                           # 物体过带轨迹(前帧点红/当前帧黄)
        (qa, qb) = band_seg
        cv2.line(plotted, (int(qa[0]), int(qa[1])),
                 (int(qb[0]), int(qb[1])), (0, 0, 255), 3)
        cv2.circle(plotted, (int(qa[0]), int(qa[1])), 5, (0, 0, 255), -1)
        cv2.circle(plotted, (int(qb[0]), int(qb[1])), 5, (0, 255, 255), -1)
        cv2.putText(plotted, "OBJ", (int(qb[0]) + 8, max(int(qb[1]) - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    mods = []
    if c["enable_climb"]: mods.append("攀爬")
    if c["enable_throw"]: mods.append("抛物")
    if c["enable_smoke"]: mods.append("抽烟")
    cv2.putText(plotted, f"MOD:{'+'.join(mods) or 'none'} FPS:{fps:.1f}",
                (W - 420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cam_label = f"CAM:{c['cam_name']}"     # 画面左上角标注当前相机(截图可溯源)
    if any(ord(ch) > 127 for ch in cam_label):
        plotted = put_text_cn(plotted, cam_label, 18, 8)
    else:
        cv2.putText(plotted, cam_label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (120, 210, 255), 2)
    if not boxes:
        cv2.putText(plotted, "no fence box (all zone)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return plotted
