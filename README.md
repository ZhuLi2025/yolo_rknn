# yolo_rknn

在 RK3588（SuperString RB1005）上跑的 YOLO 视频推理程序。拉流、硬解、预处理、NPU 推理、变焦业务分成三条线程。接上 HDMI 或 DP 并打开显示后，再加一条线程：视频层直接扫描 VPU 的 NV12，OSD 层画框。图像不走 CPU 软解，也不用 CPU 做整帧缩放和色域转换。

当前板子是 **RK3588**。需求里的 MPI、MMZ、片内 NPU 零拷贝，在这台机器上分别落到 MPP、DRM CMA 和 RKNN Runtime 2。对照表在 [本机](docs/06-this-board-vs-rv1126.md)。

## 数据怎么走

```text
RTSP/文件 --TCP，小缓冲--> FFmpeg 解封装（只拆包，不解码）
        --Annex-B--> VPU / MPP 硬解 --> NV12，DRM 物理连续内存
        --队列深度 2，满了丢旧帧--> RGA2/RGA3 缩放 + 填边 + NV12 转 RGB
        --同一张 dma-buf fd--> RKNN2 NPU --> 后处理 NMS
        batch_size=2 且 short_side：左右两张同时跑，core0 和 core1
        --> 业务线程：高度、变焦、VISCA
        --> 显示线程（可选）：同一张 NV12 进视频层，检测框进 OSD 层
```

| 线程 | 名字 | 默认绑核 | 做什么 |
| --- | --- | --- | --- |
| 拉流 + 硬解 | `yr_stream` | cpu6（A76，2.4 GHz） | 收包，送给 VPU |
| 推理 | `yr_infer` | cpu4（A76，2.35 GHz） | RGA，然后让 NPU 跑 YOLO |
| 业务 | `yr_biz` | cpu0（A55） | 高度、变焦策略、相机指令 |
| 显示 | `yr_disp` | cpu1（A55） | 仅 `display_enable=true`。视频层 + OSD 层 |

cpu4 和 cpu6 分属两个 A76 簇，收包和后处理不抢同一块 L2。

## 目录

```text
CMakeLists.txt          原生编译，C++14，不链接 libswscale
config/app.conf         运行参数
include/yolo_rknn/      对外头文件
src/                    实现
  stream/               FFmpeg 解封装
  decode/               MPP 硬解
  preprocess/           RGA，以及和显示线程共用的锁
  infer/                RKNN2 与后处理
  display/              DRM 两层显示、OSD 描边
  business/             高度、变焦、VISCA
  pipeline/             三或四条线程
tests/test_core.cpp     不依赖硬件的后处理和变焦测试
testdata/               1080p H.264 样片
docs/                   设计说明
models/                 放 rk3588 的 .rknn，不进版本库
scripts/build.sh
scripts/npu_load.sh       实时看三颗 NPU 核的占用
```

## 编译和试跑

系统编译器是 g++ 9.4，CMake 是 3.16。

```bash
./scripts/build.sh
# 不加载模型，用样片把硬解和 RGA 跑一遍
./build/bin/yolo_rknn --url testdata/sample_1080p.mp4 --decode-only
# 正式推理。模型必须是 RKNN-Toolkit2、target_platform=rk3588
./build/bin/yolo_rknn --config config/app.conf --url rtsp://192.168.1.64/live --model weights/yolov5_rk3588_int8.rknn
```

`Ctrl+C` 结束。文件播完会自己退出。看三颗 NPU 核的占用：`./scripts/npu_load.sh`（debugfs 要 root，脚本会自己要 sudo）。

默认 `display_enable=false`。本机 HDMI-A-1、HDMI-A-2、DP-1 都没接上。接上显示器后再把 `config/app.conf` 里的这一项改成 `true`。显示节点是 `/dev/dri/card0`，`card1` 是 NPU。

SCHED_FIFO 需要 root 或 `CAP_SYS_NICE`。没有这项能力时程序会打一条警告，绑核仍然生效：

```bash
sudo setcap cap_sys_nice+ep ./build/bin/yolo_rknn
```

## 文档

- [架构](docs/01-architecture.md)
- [编译与运行](docs/02-build-and-run.md)
- [拉流、硬解和内存](docs/03-memory-path.md)
- [线程](docs/04-threading.md)
- [模型与后处理](docs/05-model.md)
- [这台板子](docs/06-this-board-vs-rv1126.md)
- [从 Windows 推 RTSP](docs/07-windows-rtsp-push.md)

## 已经在本机验证过的部分

- 单元测试：`yolo_rknn_tests` 覆盖锚点头 NMS、DFL 头、变焦死区和 OSD 描边
- `testdata/sample_1080p.mp4`（H.264 1920x1080，Baseline）：VPU 解出 NV12 DRM 缓冲，RGA letterbox 到 640x640 大约 2 ms，进程在文件结束后退出
- librga 报出 RGA2 和 RGA3。NPU 驱动在 `/dev/dri/renderD129`（`fdab0000.npu`），用户态是 `librknnrt.so` 1.5.2

默认模型是 `weights/yolov5_rk3588_int8.rknn`：640 输入、1 个类别、三个 DFL 尺度。rk3399pro 的模型不能 `rknn_init`。
