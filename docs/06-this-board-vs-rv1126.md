# 这台板子和 RV1126

需求是按 RV1126 的习惯写的：MPI、MMZ、片内 NPU 零拷贝。当前内核设备树是：

```text
RK3399pro-firefly-aiojd4
rockchip,rk3399pro
```

CPU 是 Cortex-A72 ×2 + Cortex-A53 ×4。工程按这块板子能链接、能打开的设备来写。下面是需求用语和本机实现的对应关系。换成真正的 RV1126 时，不要把这份二进制直接拷过去。

## 对照

| 需求里的说法 | 本机实际 | 代码位置 |
| --- | --- | --- |
| 减小拉流缓冲，`ff_url_set_option` 的 `max_delay`、`buffer_size` | 用公开 API `av_dict_set` 写同名 AVOption，并设置 `AVFMT_FLAG_NOBUFFER` | `src/stream/rtsp_client.cpp` |
| 传输用 TCP | 默认 `transport=tcp`，UDP 必须 `allow_udp=true` 才放行 | `AppConfig::validate` |
| 禁止 CPU 软解，走 MPI / VPU，输出 NV12 | Rockchip MPP 的 `rk_mpi.h`。本机没有 rkmedia。解码类型只接受 H.264/H.265，输出强制 NV12 | `src/decode/mpp_decoder.cpp` |
| 图像放在 MMZ | `MPP_BUFFER_TYPE_DRM \| MPP_BUFFER_FLAGS_CONTIG`。内核使用 DRM CMA，日志是 `found drm allocator`。MMZ 是海思的名字 | `MppDecoder::ensure_buffer_group` |
| 预处理交给 RGA，CPU 不搬大图 | `c_RkRgaBlit`，源和目标都是 dma-buf fd。工程不链接 `libswscale` | `src/preprocess/rga_preprocessor.cpp` |
| NPU 零拷贝或低开销预处理 | RGA 之后只剩 640×640×3。RKNN 1.7.5 没有“从 fd 创建输入”的接口，`rknn_inputs_set` 会把这张张量拷进片外 NPU | `src/infer/yolov8_detector.cpp` |
| 三条线程，拉流硬解高优先级，推理和业务分开 | `yr_stream` / `yr_infer` / `yr_biz`，绑核加尝试 `SCHED_FIFO` | `src/pipeline/pipeline.cpp` |

## 为什么不能假装是 RV1126

RV1126 的应用层经常包一层 rkmedia（`RK_MPI_SYS_Init`、`VDEC` 通道那套）。那层下面仍然是 MPP。这块板子的根文件系统里没有 rkmedia 头文件和库，直接调 MPP 是能编译、也能在 `/dev/vpu_service` 上出图的路径。本机已经用 `/usr/local/test.mp4` 解出 1920×1080 NV12。

RV1126 的 NPU 在片内，RKNN2 提供 `rknn_create_mem_from_fd` 一类接口，RGA 的输出 fd 可以交给 NPU。本机头文件 `/usr/local/include/rockchip/rknn_api.h` 的版本宏是 `1.7.5`，输入结构体只有 `void* buf`。把不存在的 fd 接口写进工程，链接不过，运行也进不了这颗 NPU。

`/usr/local/lib/librknn_api.so` 甚至不是 Linux 库，而是 Android 的（`NEEDED liblog.so`，符号 `@LIBC`）。CMake 使用的是 NPU SDK 里的 Linux 构建：

```text
/home/firefly/NPU/RK3399Pro_npu/rknn-api/librknn_api/Linux/lib64/librknn_api.so
```

## 如果以后改到 RV1126

保持模块边界，只换下面几处：

1. 解码器可以继续用 MPP。若系统只提供 rkmedia，把 `MppDecoder` 换成 VDEC 通道，对外仍然交 `DecodedFrame`（fd + stride）。后面的队列和 RGA 不用改。
2. 物理连续内存如果走 MPP DRM/ION，现有 flag 多半还能用。若必须从 MMZ/CMA heap 申请，换 `ensure_buffer_group` 里的分配，不要把 malloc 的指针塞给 RGA。
3. 推理改用 RKNN2，在 `YoloV8Detector::infer` 里用 fd 导入 RGA 的目标缓冲，删掉 `rknn_inputs_set` 的整缓冲拷贝。后处理函数可以留下，但要重新核对输出 layout，Toolkit2 的 YOLOv8 头和 1.x 经常不一样。
4. 绑核改成 4 个 A7 的编号。不要沿用 cpu4/cpu5 是 A72 这个假设，`thread_util.cpp` 里的日志也要改。
5. 模型重新转换。rk3399pro 的 `.rknn` 不能在 RV1126 上 `rknn_init`。

拉流侧的 TCP、小缓冲、Annex-B、三条线程和“丢旧帧”不用因换芯片而改掉。那些限制来自实时视频，不来自具体的 NPU 型号。
