# 拉流、硬解和内存

这三件事决定 CPU 会不会被视频打满。实现都在 `src/stream`、`src/decode`、`src/preprocess`。

## 拉流

`RtspClient` 只用 FFmpeg 做解封装。

对 `rtsp://` / `rtsps://`，在 `avformat_open_input` 之前写入这些 AVOption：

| 选项 | 配置项 | 作用 |
| --- | --- | --- |
| `rtsp_transport` | `transport` | 默认 `tcp` |
| `buffer_size` | `buffer_size` | 协议层接收缓冲，默认 256 KiB |
| `max_delay` | `max_delay_us` | 解复用抖动缓冲上限，默认 100 ms |
| `reorder_queue_size` | `reorder_queue_size` | 默认 0，不做 UDP 乱序重排 |
| `stimeout` | `stimeout_us` | 套接字阻塞超时 |
| `fflags=nobuffer` | 固定 | 不要为了平滑播放去攒包 |

`AVFormatContext` 上还会设置 `AVFMT_FLAG_NOBUFFER`、`max_delay`、较小的 `probesize` 和 `max_analyze_duration`。探测不能设成 0，否则有的流还没认出编码就失败；它只影响起播，不影响稳态延迟。

这些键就是协议层的 URL 选项。旧代码里有人直接调私有符号 `ff_url_set_option`。那个符号不属于应用 API，本工程用 `av_dict_set` 写到同一组选项上。

UDP 被配置校验拒绝，除非显式写 `allow_udp=true`。丢包在这里的表现不是“画面稍微卡一下”，而是 VPU 缺参考帧，后面一串帧花屏或者解码器一直要数据。TCP 在局域网上把这个代价换成可恢复的队头阻塞，现场更稳。

非视频流会被标成 `AVDISCARD_ALL`，音频包不进解码线程。

H.264 / H.265 以外的编码直接失败，没有软解分支。MP4 里的 AVCC / HVCC 用 `h264_mp4toannexb` 或 `hevc_mp4toannexb` 转成 Annex-B 再送给 MPP。这一步拷贝的是压缩包，一帧通常几十 KB，不是整幅 NV12。

`av_read_frame` 可能堵在网络上。`AVFormatContext` 的中断回调检查停机标志，所以 `Ctrl+C` 能够把拉流线程解开。

## 硬解

`MppDecoder` 是唯一的解码入口。

初始化顺序不能颠倒：

1. `mpp_create`
2. `MPP_DEC_SET_PARSER_SPLIT_MODE = 1`（必须在 `mpp_init` 之前）。一包里带 SPS、PPS 和 IDR 时，由 VPU 自己切 NAL
3. 低延迟时再设 `MPP_DEC_SET_PARSER_FAST_MODE`（同样必须在 init 之前）
4. `mpp_init(MPP_CTX_DEC, AVC 或 HEVC)`
5. `MPP_DEC_SET_OUTPUT_FORMAT = NV12`
6. 低延迟时 `MPP_DEC_SET_IMMEDIATE_OUT`，减少显示重排序。工业相机一般是没有 B 帧的 Baseline / Main，这个开关是安全的。带 B 帧的片源可能会轻微乱序

第一帧通常是 info-change，还没有图像。这时向 MPP 提交外部 buffer group：

```text
MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG
```

申请失败时再试 `MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CONTIG`。当前用户对 `/dev/dma_heap` 没有写权限，实测走的是 DRM CMA。这里不使用 `MPP_BUFFER_TYPE_NORMAL`，那种缓冲来自普通 malloc，RGA 不能用 dma-buf fd 导入。

group 限制为 `decoder_buf_count`（默认 12）张，尺寸是 VPU 报上来的 `buf_size`。1080p 实测是 1920x1080，stride 1920x1088，一张约 4 MiB。

帧交给下游时只带 dma-buf fd、宽高和 stride。`MppFrame` 被 `shared_ptr` 握住，析构时 `mpp_frame_deinit`，buffer 才回到 VPU 的 pool。推理线程在 `improcess` 返回之后立刻丢掉这个 `shared_ptr`，不等 NPU 跑完。RGA 的 ioctl 带 `IM_SYNC`，返回时缩放结果已经在目标缓冲里。

输出格式不是线性 NV12（例如 10bit 或 AFBC）就丢帧。RK3588 的 RGA 和后面的 uint8 模型都按 8bit NV12 写。不要在这里加一条 CPU 转换。

文件结束时 `flush()` 送一个 EOS 空包，把还留在重排序里的帧取出来。直播中途不调用 `mpi->reset()`：推理线程可能还拿着上一帧的 fd。

`decode_put_packet` 返回 `MPP_ERR_BUFFER_FULL` 时，先把已经解出的帧排走再重试。码流指针只需要在这次调用期间有效；成功返回后解码器已经把压缩数据收走。

## RGA 预处理

目标是一块单独的物理连续缓冲，大小为 `model_width * model_height * 3`。分配顺序是 DRM CMA、dma-heap CMA，然后才是这两种的 cacheable 版本。非 cacheable 时 RGA 写完，NPU 可以直接读，中间不做 CPU cache 同步。

每一帧：

1. `imfill` 把目标涂成 114 灰（`0x727272`），和常见 YOLO letterbox 的填充值一致
2. `improcess` 把 NV12 按短边缩放到目标中央，坐标和宽高都对齐到偶数。源的宽高 stride 用 MPP 的 hor/ver stride，不是有效宽高
3. 只有缓冲是 cacheable 时，才用 `DMA_BUF_IOCTL_SYNC` 作废 CPU cache

librga 自己在 RGA2 和 RGA3 之间选核。模型宽高必须是 16 的倍数，否则 RGA3 会拒绝这张 RGB888。失败时返回错误字符串，调用方丢帧。没有第二套实现。

`finish_cpu_read()` 在推理返回后调用。非 cacheable 时它是空操作。预处理类不是线程安全的，只允许推理线程使用。显示线程清 OSD 时走同一把 `RgaApiLock`。

letterbox 的 `scale`、`pad_x`、`pad_y` 交给后处理，把模型坐标系的框映射回原图。缩放比用对齐之后的整数宽除以源宽，避免偶数对齐带来的半个像素误差。

## 输入怎么进 NPU

推理线程在两块硬件都初始化之后调用 `bind_input()`：

1. `rknn_create_mem_from_fd` 导入 RGA 目标缓冲的 fd
2. 输入已经是 uint8 NHWC、并且没有额外 stride 时，`pass_through=1`
3. 否则 `pass_through=0`，类型和格式写成 uint8 NHWC，由驱动做布局转换
4. `rknn_set_io_mem` 成功后，每帧只调用 `rknn_run`

导入失败时退回 `rknn_inputs_set`。640x640 RGB 大约 1.2 MiB，这一次拷贝只出现在退回路径上。全分辨率 NV12 仍然不经过 CPU。

输出仍用 `rknn_outputs_get(want_float=1)`。后处理要的是 float，反量化留在驱动里。

## 显示

`display_enable=true` 时，同一张 NV12 再进显示队列。`DrmDisplay` 用 `drmPrimeFDToHandle` + `drmModeAddFB2` 把 VPU 的 dma-buf 挂到视频平面，显示控制器自己扫描，CPU 不读这张图。当前帧和上一帧的 framebuffer 都留着，再往前的一张才 `RmFB` 并丢掉 `DecodedFrame`，避免扫描还没结束 buffer 已经被 VPU 覆写。

OSD 是另一块 dumb buffer，格式 ARGB8888。每帧先 `imfill` 涂成全透明，再 `DMA_BUF_IOCTL_SYNC` 让 CPU 只描边和写字。框的内部保持透明，所以下面的 NV12 直接透出来。两层 `SetPlane` 用同一个目标矩形做 letterbox，检测框已经是原图像素坐标，和视频对得上。

本机探测时 HDMI-A-1、HDMI-A-2、DP-1 都是断开的。没接显示器就打开显示，`init()` 会失败并让整个 pipeline 退出。显示必须打开 `/dev/dri/card0`。`card1` 是 NPU。
