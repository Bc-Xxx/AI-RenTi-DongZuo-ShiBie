# -*- coding: utf-8 -*-
"""取流(公共基础设施): 后台线程只保留最新一帧,断流自动重连。
视频文件源按自身FPS限速回放(否则读流线程以解码极限速度烧完视频,
处理线程只能抽到相隔很远的帧,运动位移过大导致检测/过带全部失效)。
任何检测模块都从这里拿画面,不自己开 VideoCapture。"""
import threading
import time

import cv2


def _open_src(src):
    # 纯数字=USB摄像头索引,走系统默认后端;rtsp/视频文件走FFMPEG(环境变量已指定tcp)
    if isinstance(src, str) and src.isdigit():
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(src, cv2.CAP_FFMPEG)


class LiveCapture:
    def __init__(self, src):
        self.src = src
        self.cap = _open_src(src)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.pace = None
        if isinstance(src, str) and not src.isdigit() \
                and not src.lower().startswith(("rtsp://", "http://", "https://")):
            vfps = self.cap.get(cv2.CAP_PROP_FPS)
            if vfps and 1 < vfps < 200:
                self.pace = 1.0 / vfps
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        nxt = time.time()
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                self.cap.release()
                time.sleep(0.5)
                if not self.running:
                    break
                self.cap = _open_src(self.src)
                nxt = time.time()
                continue
            if self.pace:                      # 按视频帧率对表,落后不追赶
                nxt += self.pace
                delay = nxt - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    nxt = time.time()
            with self.lock:
                self.frame = f
        try:                                    # 由本线程自己释放,避免与切换线程冲突
            self.cap.release()
        except Exception:
            pass

    def read(self):
        with self.lock:
            return self.frame

    def release(self):
        self.running = False                   # 只发退出信号,reader阻塞在read中时强释可能崩
