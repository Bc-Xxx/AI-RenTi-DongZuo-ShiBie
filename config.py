# -*- coding: utf-8 -*-
"""配置管理(公共基础设施): 多相机结构、旧配置自动迁移、参数合并。
与具体检测模型无关;新模块的开关/参数加进 DEFAULT_CFG 与 PARAM_KEYS 即可。"""
import json
import os
import threading

# ==================== 公共配置(集中在此,其他文件只引用不定义) ====================

# ---------- 文件路径 ----------
CFG_FILE = "web_config.json"        # 运行参数(网页滑条自动改写)
CLIMB_CFG = "climb_config.json"     # 黄框布防区标定(按相机)
SNAP_DIR = "alerts"                 # 告警截图目录
INDEX_FILE = "index.html"           # Web页面
DET_MODEL = "yolo11s.pt"            # 人/物品检测权重
POSE_MODEL = "yolo11s-pose.pt"      # 姿态权重
SMOKE_MODEL_FILE = "smoke_best.pt"  # 抽烟权重(自训,缺失自动跳过)

# ---------- COCO关键点与类别 ----------
L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10     # 左右肩/左右手腕关键点编号
ITEM_CLASSES = {24: "背包", 26: "手提包", 28: "箱"}   # 抛物持物判断用类别

# ---------- 告警通用 ----------
ALERT_LIMIT = 100                   # 告警列表最多保留条数
ALERT_COOLDOWN = 5                  # 同类告警最小间隔(秒),防连刷
SMOKING_COOLDOWN = 10               # 抽烟告警冷却(秒)
THROW_PAIR_WINDOW = 3.0             # 挥臂动作与物体过栏的配对窗口(秒)

# ---------- 抛物-过带算法常量 ----------
BAND_DIFF_TH = 25                   # 帧差二值化阈值
BAND_MASK_PAD = 15                  # 帧差抠除人框时向外扩的像素
BAND_SCENE_CUT_MEAN = 20            # 帧差均值超过=场景突变(重连/切机位),本帧不配对
BAND_MAX_BLOB_AREA = 5000           # 亮斑面积上限,超过按背景扰动丢弃
BAND_MAX_PAIR_DIST = 200            # 前后帧亮斑配对的最大位移(像素)

# ---------- 抽烟算法常量 ----------
SMOKE_MARGIN = 30                   # 烟头目标允许超出人框的像素余量

# ==================== 运行参数(存web_config.json,网页滑条可改) ====================

DEFAULT_CFG = {
    "_说明": {
        "总览": "所有参数在网页上拖动滑条会自动改写此文件；也可手动改,改完重启app.py生效。此文件不支持//注释,说明都写在_说明里,不要删",
        "rtsp": "(已废弃,保留兼容)相机地址现存在cameras列表里,网页右上角＋管理",
        "cameras": "摄像头列表(网页顶部相机标签/右上角＋管理)。每个相机独立保存:名称/取流地址/检测参数/攀爬黄框。取流地址可填rtsp地址、0(USB摄像头)或本地视频文件路径(演示用)",
        "active_cam": "当前使用的相机id。网页点相机标签即时切换,不用重启",
        "enable_climb": "攀爬检测模块开关。true开启/false关闭,网页页签也可切",
        "enable_throw": "抛物检测模块开关。true开启/false关闭",
        "enable_band": "物体过带开关(帧差捕捉飞越墙头线的物体,抛物的佐证证据)。本身不单独告警:挥臂动作+物体过栏=红色告警,仅挥臂动作=橙色预警。误报多先调band_speed,雨雪大风扬尘天气建议关闭",
        "enable_smoke": "抽烟检测模块开关。需smoke_best.pt权重文件和app.py同目录,文件缺失时模块自动跳过",
        "det_conf": "检测置信度阈值(0.25~0.7)。模型对每个目标打分,低于此值的框丢弃。误框多(把杂物框成人)→调大;远处的人/夜间的人漏检→调小。白天0.45左右,夜间可降到0.35",
        "min_person_h": "人框最小高度(像素)。比这矮的检测框直接忽略(大多是远处小误检)。设定原则:比'需要监控的最远真人在画面里的高度'小,比'常见误检杂物的高度'大。换分辨率/换机位后必须重标",
        "spike": "抛物-手腕速度阈值(像素/帧)。相邻两帧手腕移动距离超过此值=快速挥臂。正常挥手约20~40,真抛掷上百。调法:让人做两种动作看告警,卡在中间。帧率变了要重标",
        "band_min_area": "过带-运动亮斑最小面积(像素)。滤飞虫/夜视噪点。误报多→调大;小物体漏检→调小",
        "band_speed": "过带-相邻帧最小位移(像素/帧),与帧率成反比:70FPS机器约3~8,10FPS机器约15~30,以画面FPS角标为准现场标定。误报多→调大",
        "grip_frames": "攀爬-姿态持续帧数。手腕高于肩线连续这么多帧才认定攀爬姿态(单帧举手不算)。误报多→调大;反应慢→调小。6帧约0.5~1秒",
        "posture_window": "攀爬-姿态证据有效窗口(秒)。人跨坐墙头时手臂会垂下,过线发生在垂手之后;过线前N秒内出现过攀爬姿态即算证据成立。骑墙磨蹭久会漏→调大;爬完又正常走动的人被误判→调小",
        "require_posture": "攀爬-告警是否要求姿态证据。1=要求(白天推荐,误报少);0=不要求,脚底过墙头线就告警(夜间红外画面姿态不准时的兜底,宁误报不漏报)",
        "smoke_target_conf": "抽烟-目标置信度阈值(0.1~0.6)。烟头/香烟目标小且模糊,过高会漏检,过低误报多;误报靠smoke_confirm_n连续确认压制",
        "smoke_confirm_n": "抽烟-连续确认帧数。抽烟目标连续N帧关联到同一个人身上才告警。误报多→调大;检出慢→调小",
        "show_boxes": "画面上是否显示黄框/墙头线叠加。true显示/false隐藏。只影响显示,隐藏后判定照常工作"
    },
    "rtsp": "rtsp://<用户名>:<密码>@<相机IP>:554/Streaming/Channels/301",
    "enable_climb": True,
    "enable_throw": False,
    "enable_band": True,
    "enable_smoke": False,
    "det_conf": 0.45,
    "min_person_h": 50,
    "spike": 40,
    "band_min_area": 30,
    "band_speed": 15,
    "grip_frames": 6,
    "posture_window": 6,
    "require_posture": 1,
    "smoke_target_conf": 0.30,
    "smoke_confirm_n": 8,
    "show_boxes": True,
    "cameras": [],                     # [{id,name,rtsp,params:{...}}] 空则首次启动由旧配置迁移
    "active_cam": "",
}

cfg = dict(DEFAULT_CFG)
if os.path.exists(CFG_FILE):
    try:
        cfg.update(json.load(open(CFG_FILE, encoding="utf-8")))
    except Exception as e:
        print(f"配置读取失败用默认: {e}")

cfg_lock = threading.RLock()           # RLock: merged_view/save_cfg可在持锁状态内嵌套调用

# 参数归属划分:PARAM_KEYS每个相机独立存一份(不同机位场景不同,阈值/标定必须各调各的),
# 模块开关等其余键为全局。旧版单相机配置(参数在顶层)首次启动自动迁入cameras[0]。
PARAM_KEYS = ["det_conf", "min_person_h", "spike", "band_min_area", "band_speed",
              "grip_frames", "posture_window",
              "require_posture", "smoke_target_conf", "smoke_confirm_n"]


def gen_cam_id(used=None):
    used = set(used if used is not None else (c["id"] for c in cfg["cameras"]))
    i = 1
    while f"cam{i}" in used:
        i += 1
    return f"cam{i}"


def normalize_cfg():
    """旧版(单相机/参数在顶层) -> 多相机结构,幂等"""
    cams = cfg.get("cameras") or []
    if not cams:
        cams = [{"id": "cam1", "name": "相机1",
                 "rtsp": cfg.get("rtsp", DEFAULT_CFG["rtsp"]),
                 "params": {k: cfg.get(k, DEFAULT_CFG[k]) for k in PARAM_KEYS}}]
    seen = set()
    for c in cams:
        if not c.get("id") or c["id"] in seen:
            c["id"] = gen_cam_id(seen)
        seen.add(c["id"])
        c.setdefault("params", {})
        for k in PARAM_KEYS:
            c["params"].setdefault(k, cfg.get(k, DEFAULT_CFG[k]))
        c["name"] = str(c.get("name") or c["id"])
        c["rtsp"] = str(c.get("rtsp") or "")
    cfg["cameras"] = cams
    if cfg.get("active_cam") not in seen:
        cfg["active_cam"] = cams[0]["id"]
    for k in PARAM_KEYS + ["rtsp"]:    # 顶层旧键已迁走,删掉避免两处存储打架
        cfg.pop(k, None)
    sh = cfg.setdefault("_说明", {})
    for key in ("cameras", "active_cam"):
        sh.setdefault(key, DEFAULT_CFG["_说明"][key])


normalize_cfg()


def save_cfg():
    with cfg_lock:
        json.dump(cfg, open(CFG_FILE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)


def active_cam_obj():
    for c in cfg["cameras"]:
        if c["id"] == cfg["active_cam"]:
            return c
    return cfg["cameras"][0]


def merged_view():
    """全局开关+当前相机参数的扁平视图(引擎和/api/config共用)"""
    with cfg_lock:
        c = active_cam_obj()
        out = {k: cfg.get(k, DEFAULT_CFG[k])
               for k in ("enable_climb", "enable_throw", "enable_band",
                         "enable_smoke", "show_boxes")}
        out.update(c["params"])
        out["rtsp"] = c["rtsp"]
        out["active_cam"] = c["id"]
        out["cam_name"] = c["name"]
        return out


save_cfg()                             # 结构升级立即落盘,下次启动直接是新格式
