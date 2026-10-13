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

这是本机的物理连续内存。MPP 日志里会写 `found drm allocator`（没有 ION）。海思文档里的 MMZ 就是同一类东西：给硬件用的、物理地址连续的专用内存。这里不使用 `MPP_BUFFER_TYPE_NORMAL`，那种缓冲来自普通 malloc，RGA 不能用 dma-buf fd 导入。

group 限制为 `decoder_buf_count`（默认 12）张，尺寸是 VPU 报上来的 `buf_size`。1080p 实测是 1920x1080，stride 1920x1088，一张约 4 MiB。

帧交给下游时只带 dma-buf fd、宽高和 stride。`MppFrame` 被 `shared_ptr` 握住，析构时 `mpp_frame_deinit`，buffer 才回到 VPU 的 pool。推理线程在 `c_RkRgaBlit` 返回之后立刻丢掉这个 `shared_ptr`，不等 NPU 跑完。RGA 的 ioctl 是同步的，返回时缩放结果已经在目标缓冲里。

输出格式不是 NV12（例如 10bit）就丢帧。本机 RGA 和后面的 uint8 模型都按 8bit NV12 写。不要在这里加一条 CPU 转换。

文件结束时 `flush()` 送一个 EOS 空包，把还留在重排序里的帧取出来。直播中途不调用 `mpi->reset()`：推理线程可能还拿着上一帧的 fd。

`decode_put_packet` 返回 `MPP_ERR_BUFFER_FULL` 时，先把已经解出的帧排走再重试。码流指针只需要在这次调用期间有效；成功返回后解码器已经把压缩数据收走。

## RGA 预处理

目标是一块单独的 DRM 缓冲，大小为 `model_width * model_height * 3`，优先 `CONTIG | CACHABLE`。申请不到 cacheable 时退回只要求物理连续，并打警告。

每一帧：

1. `c_RkRgaColorFill` 把目标涂成 114 灰（`0x00727272`），和常见 YOLO letterbox 的填充值一致
2. `c_RkRgaBlit` 把 NV12 按短边缩放到目标中央，坐标和宽高都对齐到偶数。源的 `wstride/hstride` 用 MPP 的 hor/ver stride，不是有效宽高
3. `DMA_BUF_IOCTL_SYNC` 以 `DMA_BUF_SYNC_READ | START` 作废 CPU cache

第 3 步是因为 RGA 通过 DMA 写内存。buffer 如果是 cacheable 的，CPU 随后读到的可能还是旧 cache。`rknn_inputs_set` 会从用户态指针把这张 RGB 拷进 NPU，拷贝之前必须看到新数据。

`finish_cpu_read()` 在 `rknn_inputs_set` 返回后调用，结束这次 CPU 读同步。指针在下一次 `process()` 之前有效，预处理类不是线程安全的，只允许推理线程使用。

letterbox 的 `scale`、`pad_x`、`pad_y` 交给后处理，把模型坐标系的框映射回原图。缩放比用对齐之后的整数宽除以源宽，避免偶数对齐带来的半个像素误差。

源 fd 和目标 fd 都设 `mmuFlag = 1`，这是 RK3399 上 dma-buf 导入 RGA 的用法。失败时返回错误码，调用方丢帧。没有第二套实现。

## 仍然会发生的那一次拷贝

RKNN API 1.7.5 的输入接口是 `rknn_inputs_set(void* buf)`。RK3399Pro 的 NPU 是片外协处理器，驱动必须把输入张量送过去。640x640 RGB 大约 1.2 MiB。

全分辨率 NV12（1080p 约 3 MiB，4K 更大）不经过 CPU。被禁止的是用 CPU 搬运和转换这一档数据。模型输入这一档拷贝是这套 API 和这颗 NPU 的边界，不是漏做的优化。RV1126 片内 NPU 加上 RKNN2 的 fd 输入才能把这一档也去掉，接口见 [对照说明](06-this-board-vs-rv1126.md)。

## 显示

`display_enable=true` 时，同一张 NV12 再进显示队列。`DrmDisplay` 用 `drmPrimeFDToHandle` + `drmModeAddFB2` 把 VPU 的 dma-buf 挂到视频平面，显示控制器自己扫描，CPU 不读这张图。当前帧和上一帧的 framebuffer 都留着，再往前的一张才 `RmFB` 并丢掉 `DecodedFrame`，避免扫描还没结束 buffer 已经被 VPU 覆写。

OSD 是另一块 dumb buffer，格式 ARGB8888。每帧先 `c_RkRgaColorFill` 涂成全透明，再 `DMA_BUF_IOCTL_SYNC` 让 CPU 只描边和写字。框的内部保持透明，所以下面的 NV12 直接透出来。两层 `SetPlane` 用同一个目标矩形做 letterbox，检测框已经是原图像素坐标，和视频对得上。

本机探测时 HDMI-A-1 是断开的（`connection=2`，没有 mode）。没接显示器就打开显示，`init()` 会失败并让整个 pipeline 退出，日志里有 `HDMI-A-1 is unplugged`。
