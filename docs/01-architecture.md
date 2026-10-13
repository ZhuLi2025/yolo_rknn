# 架构

程序是一个进程。`display_enable=false` 时是三条线程、两级队列；接上 HDMI 并把显示打开后，再加一条显示线程和两级队列。线程之间不共享 MPP context，也不共享 RKNN context。这两种句柄都不是线程安全的。librga 同样不能并发，推理线程和显示线程共用一把进程内锁。

```text
主线程
  读配置、注册 SIGINT/SIGTERM、start()、wait()
  信号处理器只把 atomic 停机标志写上，不在信号里 join

yr_stream
RtspClient 解封装
MppDecoder 硬解
        |
        +---- BoundedQueue<DecodedFrame> ---- yr_infer
        |                                      RgaPreprocessor
        |                                      YoloV8Detector
        |                                          |
        |                                          +-- BoundedQueue<InferenceResult> -- yr_biz
        |                                          |                                    高度 / 变焦 / VISCA
        |                                          +-- BoundedQueue<InferenceResult> -- yr_disp（OSD）
        |
        +---- BoundedQueue<DecodedFrame> ---- yr_disp
                                              视频层扫 NV12，OSD 层画框
```

## 模块边界

| 模块 | 允许做的事 | 明确不做的事 |
| --- | --- | --- |
| `RtspClient` | `av_read_frame`，bitstream filter 转 Annex-B | `avcodec_open2`，把像素解出来 |
| `MppDecoder` | `mpp_create` / `decode_put_packet` / `decode_get_frame` | 软解，把 NV12 拷到普通堆内存 |
| `RgaPreprocessor` | `c_RkRgaBlit`，fd 对 fd | `libswscale`，CPU resize，CPU `cvtColor` |
| `YoloV8Detector` | `rknn_inputs_set` / `rknn_run` | 自己遍历 NV12 |
| `decode_yolov8` | 对模型输出张量做解码和 NMS | 接触图像 |
| `Controller` | 读高度、算变焦、写 VISCA | 碰解码器和 NPU |
| `DrmDisplay` | 视频层 `drmModeSetPlane` 扫 NV12，OSD 层画框 | CPU 清整帧、CPU 做 NV12 转 RGB |
| `osd_draw` | 在已经清成透明的 ARGB 上描边、写字 | 填满检测框内部 |

后处理、变焦策略和 OSD 描边是纯函数。`tests/test_core.cpp` 不链接 MPP / RGA / RKNN / DRM。

## 队列语义

四个队列默认深度都是 2，满了丢最旧的元素。拉流线程的 `push` 不会阻塞。推理或显示慢的时候，VPU 仍然持续把最新帧解出来，旧帧连同它占着的 DRM buffer 一起释放。同一帧的 `DecodedFrame` 可以同时待在推理队列和显示队列里，引用计数归零才把 buffer 还给 VPU。

这是实时视频的取舍：要的是最新画面，不是把每一帧都推理完。框会比当前画面晚一次推理周期。丢帧计数在 `stream ... queue_drop=` 那条日志里。

## 生命周期

1. `start()` 拉起工作线程，最多等 30 秒。显示关闭时等 3 条，打开时等 4 条。任何一条初始化失败，其余线程会被停机标志打断，然后 join。
2. 解码器对象在拉流线程里创建，但用 `shared_ptr` 交回 `Pipeline`。推理线程或显示线程还拿着 NV12 帧的时候不能销毁 MPP buffer group。
3. `stop()` 的顺序是：等拉流线程退出，关闭两路帧队列，等推理线程和显示线程退出，清空残余帧，销毁解码器，再关闭结果队列并等业务线程。顺序反了就会在 RGA 或显示控制器还在用 fd 的时候把 DRM buffer 收走。

## 失败时的态度

硬解输出不是 NV12、RGA blit 失败、模型输出布局对不上，都是丢掉这一帧并打日志。代码里没有 CPU 后备路径。加上后备路径之后，现场一旦 RGA 或 VPU 异常，CPU 会被整帧图像打满，这正是要避免的情况。
