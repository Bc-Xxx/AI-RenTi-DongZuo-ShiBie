# -*- coding: utf-8 -*-
"""Web服务层: FastAPI全部接口(页面/参数配置/摄像头管理/黄框标定/告警/MJPEG推流)。
只做HTTP编排,业务逻辑在config/zones/alerts/capture。"""
import os
import threading
import time

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import alerts
import capture
import config
import zones

app = FastAPI()


class CfgIn(BaseModel):
    enable_climb: bool | None = None
    enable_throw: bool | None = None
    enable_band: bool | None = None
    enable_smoke: bool | None = None
    det_conf: float | None = None
    min_person_h: int | None = None
    spike: float | None = None
    band_min_area: int | None = None
    band_speed: float | None = None
    grip_frames: int | None = None
    posture_window: float | None = None
    require_posture: int | None = None
    smoke_target_conf: float | None = None
    smoke_confirm_n: int | None = None
    show_boxes: bool | None = None


@app.get("/")
def index():
    return FileResponse(config.INDEX_FILE, media_type="text/html")


@app.get("/api/config")
def get_config():
    return config.merged_view()


@app.post("/api/config")
def set_config(v: CfgIn):
    with config.cfg_lock:
        ac = config.active_cam_obj()
        for k, val in v.dict().items():
            if val is None:
                continue
            if k in config.PARAM_KEYS:     # 阈值类参数跟随当前相机存储
                ac["params"][k] = val
            else:
                config.cfg[k] = val
    config.save_cfg()
    return {"ok": True}


# ---------------- 摄像头管理 ----------------
class CamIn(BaseModel):
    id: str | None = None              # 新增相机可不带,服务端分配
    name: str = ""
    rtsp: str = ""


class CamListIn(BaseModel):
    cameras: list[CamIn]


@app.get("/api/cameras")
def list_cams():
    with config.cfg_lock:
        return {"active": config.cfg["active_cam"],
                "cameras": [{"id": c["id"], "name": c["name"], "rtsp": c["rtsp"]}
                            for c in config.cfg["cameras"]]}


@app.post("/api/cameras")
def save_cams(v: CamListIn):
    """整表保存(网页摄像头管理弹窗)。已有id的相机继承原参数与黄框,新相机用默认参数"""
    if not v.cameras:
        raise HTTPException(400, "至少保留一个相机")
    with config.cfg_lock:
        old = {c["id"]: c for c in config.cfg["cameras"]}
        used, cams = set(), []
        for i, c in enumerate(v.cameras):
            cid = (c.id or "").strip()
            if not cid or cid in used:
                cid = config.gen_cam_id(used | set(old))
            used.add(cid)
            base = old.get(cid)
            cams.append({"id": cid,
                         "name": (c.name or "").strip() or f"相机{i + 1}",
                         "rtsp": (c.rtsp or "").strip(),
                         "params": dict(base["params"]) if base
                         else {k: config.DEFAULT_CFG[k] for k in config.PARAM_KEYS}})
        if config.cfg["active_cam"] not in used:      # 当前相机被删:自动落到第一个
            config.cfg["active_cam"] = cams[0]["id"]
        config.cfg["cameras"] = cams
        active_after = config.cfg["active_cam"]
        config.save_cfg()
    zones.drop_cam_boxes(used)             # 顺手清掉被删相机的黄框
    return {"ok": True, "active": active_after}


class ActiveIn(BaseModel):
    id: str


@app.post("/api/cameras/active")
def set_active_cam(v: ActiveIn):
    with config.cfg_lock:
        for c in config.cfg["cameras"]:
            if c["id"] == v.id:
                config.cfg["active_cam"] = v.id
                config.save_cfg()
                return {"ok": True}
    raise HTTPException(404, "相机不存在")


@app.get("/api/cameras/test")
def test_camera(rtsp: str):
    """测试取流地址能否读到画面(8秒超时,后台线程探测,地址错时不卡界面)"""
    res = {"ok": None}

    def _probe():
        try:
            cap = capture._open_src(rtsp)
            ok, f = cap.read()
            cap.release()
            res["ok"] = bool(ok and f is not None)
        except Exception:
            res["ok"] = False

    threading.Thread(target=_probe, daemon=True).start()
    t0 = time.time()
    while res["ok"] is None and time.time() - t0 < 8:
        time.sleep(0.1)
    if res["ok"] is None:
        return {"ok": False, "msg": "超时(>8秒),检查地址/网络/相机在线"}
    return {"ok": res["ok"],
            "msg": "连接成功,可读到画面" if res["ok"] else "打不开或读不到帧,检查地址与流子码流"}


@app.get("/api/alerts")
def get_alerts():
    with alerts.out_lock:
        return list(alerts.ALERTS)


@app.get("/api/status")
def status():
    with alerts.out_lock:
        age = time.time() - alerts.LAST_FRAME_TS
    with config.cfg_lock:
        ac = config.active_cam_obj()
    return {"alive": age < 3.0, "fps_age": round(age, 2),
            "cam": ac["id"], "cam_name": ac["name"]}


@app.get("/alerts/{name}")
def alert_img(name: str):
    p = os.path.join(alerts.SNAP_DIR, os.path.basename(name))
    if not os.path.exists(p):
        return Response(status_code=404)
    return FileResponse(p, media_type="image/jpeg")


class BoxesIn(BaseModel):
    boxes: list
    append: bool = False       # True=追加到已有框(画拐角第二面墙), False=整体替换


@app.get("/api/boxes")
def get_boxes():
    return {"boxes": [b["pts"] for b in zones.BOXES]}


@app.post("/api/boxes")
def set_boxes(v: BoxesIn):
    """网页画完黄框:写进当前相机在climb_config.json里的条目并热加载"""
    with config.cfg_lock:
        cam_id = config.cfg["active_cam"]
    count = zones.save_cam_boxes(cam_id, v.boxes, v.append)
    return {"ok": True, "count": count}


def mjpeg():
    while True:
        with alerts.out_lock:
            j = alerts.LATEST_JPEG
        if j:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
        time.sleep(0.05)


@app.get("/api/stream")
def stream():
    return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")
