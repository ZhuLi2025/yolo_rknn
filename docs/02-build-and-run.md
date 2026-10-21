# 编译与运行

## 环境

这些版本是在本机上读到的，工程按它们编写：

| 组件 | 版本或路径 |
| --- | --- |
| 板子 | SuperString RK3588 RB1005，内核 5.10.160，8 核 |
| 编译器 | g++ 9.4.0，C++14 |
| CMake | 3.16.3 |
| FFmpeg | libavformat 58.45.100，装在 `/usr/lib/aarch64-linux-gnu` |
| MPP | `librockchip_mpp.so`，头文件 `/usr/include/rockchip`，包版本 20230628 |
| RGA | `librga.so` 2.2.0，头文件 `/usr/include/rga`。运行时报告 RGA2 和 RGA3，API v1.9.2 |
| RKNN | Runtime 2，`/usr/lib/librknnrt.so` 1.5.2，头文件 `/usr/include/rknn_api.h` |

用户 `ubuntu` 对 `/dev/dri/card0`、`renderD128`、`renderD129` 有 ACL，能够打开显示节点和 NPU。`/dev/mpp_service` 和 `/dev/rga` 对其他用户也可写。`/dev/dma_heap/*` 只给 `video` 组写，当前用户不在这个组里，所以缓冲分配走 DRM CMA。把用户加进 `video` 组之后，dma-heap 那条后备路径才会成功。

## 编译

```bash
./scripts/build.sh
```

脚本会配置 `build/`、编译、并跑 `yolo_rknn_tests`。产物：

- `build/bin/yolo_rknn`
- `build/bin/yolo_rknn_tests`

手动编译：

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

工程有意不链接 `libswscale`。

## 运行

配置文件是 `config/app.conf`，格式为 `key=value`。在仓库根目录启动时，如果没有写 `--config`，会自动读这个文件。命令行再覆盖 URL 和模型。

```bash
# 只验证拉流、VPU、RGA
./build/bin/yolo_rknn --url testdata/sample_1080p.mp4 --decode-only

# RTSP。传输协议默认 tcp，max_delay 和 buffer_size 来自配置
./build/bin/yolo_rknn --url rtsp://192.168.1.64/live --model models/yolov8n.rknn
```

| 参数 | 作用 |
| --- | --- |
| `--config FILE` | 指定配置文件。文件不存在就直接失败 |
| `--url URL` | 覆盖 `url`。支持 `rtsp://` 和本地媒体文件 |
| `--model FILE` | 覆盖 `model_path` |
| `--decode-only` | 不加载 NPU。RGA 仍然会跑，用来确认预处理没有掉回 CPU |
| `-h` | 帮助 |

退出：

- `SIGINT` / `SIGTERM`：主线程看见停机标志后 join 三条线程
- 本地文件读到 EOF：拉流线程刷完 VPU 里残留的帧，置停机标志，进程退出
- RTSP 读失败：等 1 秒重连，不重置正在被推理线程使用的解码器。连续 30 次硬解失败则整个进程停下来

## 日志

写到标准错误，一行一条，带级别、时间和源文件行号。`log_level` 可以是 `debug`、`info`、`warn`、`error`。

正常硬解时能看到类似：

```text
video h264 1920x1080 stream=0 (demux only, no software decoder)
VPU decoder ready codec=h264 low_latency=1
NV12 DRM buffers 1920x1080 stride 1920x1088 size=4177920 count=12
seq=... det=0 height=10.00m preprocess=4.7ms infer=0.0ms
```

`preprocess` 是 RGA 的耗时。`infer` 在 `--decode-only` 下是 0。加载模型之后，`infer` 是 `rknn_run` 到 `rknn_outputs_get` 的时间。输入走 fd 时，这段时间里没有把 640x640 RGB 从用户态再拷一遍。退回 `rknn_inputs_set` 时，拷贝算在 `infer` 里。

## 调度优先级

绑核不需要 root。`SCHED_FIFO` 需要 `CAP_SYS_NICE`。失败时日志是：

```text
SCHED_FIFO 50 not applied (1). Need CAP_SYS_NICE or root; thread isolation still holds via affinity
```

```bash
sudo setcap cap_sys_nice+ep ./build/bin/yolo_rknn
```

优先级数字见配置里的 `stream_priority`、`infer_priority`、`business_priority`，范围 1–99，0 表示不改调度策略。
