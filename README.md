# AI人体动作识别

基于 YOLO11 的工厂围栏安防检测系统：**攀爬翻越围栏检测**、**抛掷物品检测**、**抽烟违规检测** 三大模块 + 本地 Web 管理平台（实时画面 / 模块开关 / 参数在线调节 / 告警截图）。

设计目标：部署在边缘计算盒（NVIDIA Jetson Orin NX）上，接入工厂 RTSP 监控相机，7×24 小时实时检测与告警。

## 功能模块

| 模块 | 判定逻辑 | 告警级别 |
|---|---|---|
| 攀爬翻越 | 黄框布防区 + 脚底轨迹穿过围栏顶线（几何判定）+ 时间窗口内攀爬姿态证据（手腕持续高于肩） | 预警 / 告警 |
| 抛掷物品 | 布防区内人员 + 手腕速度尖峰（快速挥臂）+ 手持物关联（背包/手提包/箱类） | 告警 |
| 抽烟违规 | 烟头/吸烟目标检测 + 连续 N 帧关联到同一人 | 告警 |

多证据投票设计：单一证据不告警，多项证据对上才升级为正式告警，压误报。

## 文件说明

| 文件 | 说明 |
|---|---|
| `app.py` | Web 平台主程序（FastAPI + 检测引擎），`python app.py` 启动 |
| `index.html` | Web 界面（实时画面、模块开关、参数滑条、告警列表、画布防区） |
| `web_config.json` | 参数配置（含每个参数的中文说明，网页调节自动保存） |
| `climb_config.json` | 黄框布防区标定（网页上鼠标绘制） |
| `climb_demo.py` | 攀爬检测单机版（命令行调试用） |
| `throw_demo.py` | 抛物检测单机版 |
| `rtsp_test.py` | RTSP 相机接入测试（含低延迟取流线程） |
| `smoking_demo.py` | 抽烟检测单机版 |

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+）
pip install ultralytics fastapi uvicorn

# 2. 下载预训练权重放到本目录（首次运行也会自动下载）
#    https://github.com/ultralytics/assets/releases  下载 yolo11s.pt 和 yolo11s-pose.pt
#    抽烟模块需自训权重 smoke_best.pt（缺失时该模块自动跳过）

# 3. 修改 web_config.json 里的 rtsp 为你的相机地址

# 4. 启动，浏览器打开 http://localhost:8000
python app.py
```

## 使用流程

1. 顶栏页签开关三个检测模块，齿轮图标进入各模块参数设置
2. 攀爬设置 → “绘制/重画黄框” → 在画面上点击圈住围栏（**前两个点决定墙头触发线**），右键撤销，双击完成
3. 滑条调参即时生效并持久化；告警自动截图存 `alerts/` 目录
4. 首次部署需现场标定：置信度阈值、人框最小高度、手腕速度阈值等（说明见 `web_config.json` 的 `_说明` 字段）

## 部署说明

- 开发调试：Windows + CPU 可运行；正式部署：Jetson Orin NX（建议导出 TensorRT engine 加速，`yolo export model=xx.pt format=engine`）
- 相机建议：RTSP over TCP（代码已内置），围栏点位建议主码流 1080p
- 夜间红外画面姿态证据可靠性下降，可将 `require_posture` 设为 0 切换为纯几何判定兜底

## 注意事项

- 相机账号密码等敏感信息请只写在本地 `web_config.json`，**不要提交到仓库**（本项目模板中已替换为占位符）
- ultralytics 为 AGPL-3.0 协议，本地部署使用无开源义务，封装为对外商业 SaaS 需注意授权
