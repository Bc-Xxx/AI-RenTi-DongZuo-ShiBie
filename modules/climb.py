# -*- coding: utf-8 -*-
"""攀爬检测规则(纯规则,可单测):
黄框内脚底轨迹穿过墙头线(几何) + 攀爬姿态时间窗口证据 → 两级判定
  CLIMB!   告警: 过线 + (过线前posture_window秒内出现过持续攀爬姿态 或 不要求姿态)
  CLIMBING 预警: 框内当前出现持续攀爬姿态
姿态证据(grip_now)由 evidence.pose_person 计算,本模块只做状态推进与判定。"""
from zones import cross_any_top


def step(st, box, feet, cur_in, prev_in, grip_now, c, t_now, labels):
    """单人多帧状态推进 → (cross_alarm过线告警, climb_warn姿态预警)

    st: 该追踪id的多帧状态字典(grip/last_climb/prev_feet由本模块维护)
    box: (x1,y1,x2,y2) 人框; feet: 脚底点; c: merged_view参数视图; labels: 画面标签列表"""
    x1, y1, _, _ = box
    st["grip"] = st["grip"] + 1 if grip_now else 0
    if st["grip"] >= c["grip_frames"]:
        st["last_climb"] = t_now                 # 记住最近一次攀爬姿态的时刻
    # 跨坐墙头时手臂会垂下,姿态证据与过线不同时发生:
    # 改为"过线前 POSTURE_WINDOW 秒内出现过持续攀爬姿态"即有效
    climbing_recent = (t_now - st["last_climb"]) < c["posture_window"]
    posture_ok = climbing_recent or not c["require_posture"]

    cross_alarm = climb_warn = False
    if cross_any_top(st["prev_feet"], feet) and (prev_in or cur_in) and posture_ok:
        labels.append((x1, y1, "CLIMB!", (0, 0, 255)))
        cross_alarm = True
    elif cur_in and st["grip"] >= c["grip_frames"]:
        labels.append((x1, y1, "CLIMBING", (0, 200, 255)))
        climb_warn = True
    return cross_alarm, climb_warn
