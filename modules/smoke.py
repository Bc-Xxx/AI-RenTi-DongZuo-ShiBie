# -*- coding: utf-8 -*-
"""抽烟检测规则(纯规则,可单测):
烟头/吸烟目标(evidence.smoke_targets检出) 连续N帧空间关联到同一个人 → SMOKING。
连续确认帧数是压误报的关键:烟头目标小且模糊,单帧误检多,连续N帧都
误检在同一人身上的概率极低。目标检出在evidence层,本模块只做关联与计数。"""
from config import SMOKE_MARGIN


def person_step(st, box, smoke_targets, c, labels):
    """单人多帧状态推进(st["smoke"]确认计数) → 本帧此人是否达到告警确认"""
    x1, y1, x2, y2 = box
    if not smoke_targets:
        return False                     # 本帧无检出目标,确认计数冻结(原实现语义)
    hit = any(x1 - SMOKE_MARGIN <= tcx <= x2 + SMOKE_MARGIN and
              y1 - SMOKE_MARGIN <= tcy <= y2 + SMOKE_MARGIN
              for tcx, tcy, *_ in smoke_targets)
    st["smoke"] = st["smoke"] + 1 if hit else 0
    if st["smoke"] >= c["smoke_confirm_n"]:
        labels.append((x1, y1, "SMOKING", (200, 80, 255)))
        return True
    return False
