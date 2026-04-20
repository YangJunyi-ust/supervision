# 拉流人流识别流程图（RTMP）

对应脚本：`rtmp_stream.py`（复用 `ultralytics_example.py` 中的检测与标注逻辑）。

在 VS Code、Cursor、GitHub 等支持 [Mermaid](https://mermaid.js.org/) 的预览中打开本文件即可渲染流程图。

---

## 1. 入口：两种使用方式

```mermaid
flowchart TB
    Start([开始]) --> Mode{是否指定<br/>capture_frame_to?}
    Mode -->|是| C1[连接 RTMP<br/>重试打开流]
    C1 --> C2[读一帧]
    C2 --> C3[保存图片并退出]
    Mode -->|否| Z{是否提供<br/>zone_configuration_path?}
    Z -->|否| Err[报错退出]
    Z -->|是| Run[进入 run：实时识别与展示]
```

---

## 2. 实时拉流（`run`）主流程

```mermaid
flowchart TB
    subgraph Init[初始化]
        A1[读取 zones JSON]
        A2[加载 YOLO 权重]
        A3[连接 RTMP 可重试]
        A4[按分辨率创建区域与标注器]
        A5[启动后台推理线程]
    end

    subgraph Worker[后台推理线程]
        W1{队列中有帧?}
        W2[取最新一帧<br/>队列最多 1 帧]
        W3[人体检测 detect]
        W4[更新 latest_detections]
        W1 -->|否| Wsleep[短暂等待]
        Wsleep --> W1
        W1 -->|是| W2 --> W3 --> W4 --> W1
    end

    subgraph Main[主线程：拉流与显示]
        M1[读取一帧]
        M2{读帧成功?}
        M3[断流重连]
        M4{到达 frame_stride?}
        M5[拷贝帧入推理队列]
        M6[annotate 分区画框与人数]
        M7[窗口显示]
        M8{按 q 退出?}
        M9[结束]
        M1 --> M2
        M2 -->|否| M3 --> M1
        M2 -->|是| M4
        M4 -->|是| M5
        M4 -->|否| M6
        M5 --> M6
        M6 --> M7 --> M8
        M8 -->|否| M1
        M8 -->|是| M9
    end

    Init --> Main
    Init --> Worker
    M9 --> Clean[停止线程、释放资源、关闭窗口]
```

---

## 3. 检测与标注的数据流

```mermaid
flowchart LR
    F[视频帧] --> Q[推理队列<br/>仅保留最新 1 帧]
    Q --> D[YOLO 检测<br/>可选 SAHI 分块]
    D --> R[检测结果]
    R --> Ann[按 PolygonZone 过滤并标注]
    ZN[区域多边形配置] --> Ann
    F --> Ann
    Ann --> Out[带区域与人数的画面]
```

---

## 说明摘要

| 环节 | 作用 |
|------|------|
| 推理队列 `deque(maxlen=1)` | 只处理最新帧，避免推理积压导致延迟越来越大 |
| `frame_stride` | 每 N 帧送检一次，降低算力；中间帧沿用上次检测结果 |
| 双线程 | 主线程负责读流与显示，推理线程负责 `detect`，减少卡顿感 |
| 断流重连 | `read` 失败时按配置重试打开流 |
