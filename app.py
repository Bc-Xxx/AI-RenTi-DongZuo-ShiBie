# -*- coding: utf-8 -*-
"""
安防检测平台 入口：攀爬/抛物/抽烟检测 + 多摄像头Web管理
运行:
  pip install ultralytics fastapi uvicorn
  python app.py
浏览器打开 http://localhost:8000

代码结构(各文件职责见其文件头):
  app.py      入口:拉起引擎线程与Web服务(本文件)
  config.py   配置/多相机/参数        capture.py  取流
  zones.py    黄框标定+几何           alerts.py   告警与共享帧
  evidence.py 模型推理(证据计算)      viz.py      画面叠加绘制
  engine.py   引擎主循环(编排)        web.py      FastAPI接口
  modules/    检测模块规则(纯判定,可单测;新模型接入说明见其__init__.py)
"""
import threading

import uvicorn
from web import app
from engine import engine

if __name__ == "__main__":
    threading.Thread(target=engine, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
