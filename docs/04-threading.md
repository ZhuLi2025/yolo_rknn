# 线程

RK3399Pro 有 2 个 Cortex-A72（cpu4、cpu5）和 4 个 Cortex-A53（cpu0–cpu3）。它不是一颗扛不住软解的单核 A7，但 1080p/4K 软解仍然会把内存带宽和 CPU 占满，所以软解这条路在代码里不存在。绑核是为了让收包和送 NPU 不要抢同一颗核，也避免业务线程的串口和策略计算插进拉流线程。

## 分工

### yr_stream，默认 cpu4，SCHED_FIFO 50

只做两件事：从 `RtspClient::read` 拿 Annex-B，调用 `MppDecoder::decode`。MPP 的 put/get 都在这一个线程里。不要再拆一个“只收包”的线程把码流队列做深，那会把延迟堆回去，也让 MPP context 跨线程。

队列满时在这个线程里丢掉最旧的 `DecodedFrame`。丢弃发生在队列锁外面，`mpp_frame_deinit` 不会占着队列锁。

### yr_infer，默认 cpu5，SCHED_FIFO 30

创建 `RgaPreprocessor` 和 `YoloV8Detector`。循环：

1. 取出一帧 NV12
2. RGA letterbox 到模型输入
3. 立刻释放 NV12，把 DRM buffer 还给 VPU
4. `rknn_inputs_set` + `rknn_run` + `rknn_outputs_get`
5. 结束 dma-buf 的 CPU 读同步
6. 把 `InferenceResult` 推进业务队列

NPU 在跑的时候，这颗 A72 大部分时间在等驱动返回。把它和拉流线程分开，是避免 `rknn_inputs_set` 里的拷贝卡住收包。

`--decode-only` 时这一线程仍然做第 2 步。跳过 RGA 就看不出预处理有没有掉回 CPU。

### yr_disp，默认 cpu1，SCHED_FIFO 15

只有 `display_enable=true` 才创建。循环里先取一张 NV12 交给 `present()`，再把 OSD 队列里积压的结果都画掉，屏幕上留下最新的一组框。视频按解码节奏刷新，框按推理节奏刷新，两者不必是同一帧。

这一线程和推理线程都会进 librga。`RgaApiLock` 把 color fill 和 blit 串起来。显示线程的优先级低于业务线程，避免清屏 ioctl 抢在变焦指令前面。

### yr_biz，默认 cpu0，SCHED_FIFO 20

每 50 ms 醒一次。有新的检测结果就更新，然后读高度、计算变焦、必要时写串口。没有检测结果时也读传感器，但没有目标就不会转动镜头。

## 变焦

`decide_zoom` 是纯函数，逻辑写在 `src/business/zoom_policy.cpp`。

有目标时：

```text
error = target_bbox_ratio - (框高 / 画面高)
gain  = clamp(当前高度 / 参考高度, 0.5, 2)    # 没有高度时 gain = 1
score = error * gain
```

`score` 大于 `target_bbox_ratio * zoom_deadband` 就放大，小于负的死区就缩小，否则保持。高度只改变增益。误差为 0 时，飞得再高结果仍是保持，镜头不会一直转到头。

没有目标时直接保持。用高度去做开环放大，停不下来。

指令有冷却时间 `zoom_cooldown_ms`，避免在死区边缘来回发令。从放大/缩小回到保持时，冷却不生效，马上发一次 VISCA 停止（`81 01 04 07 00 FF`），否则镜头会停不下来。放大是 `81 01 04 07 02 FF`，缩小是 `81 01 04 07 03 FF`，地址固定为 1。

`camera_device` 为空时，这些字节只出现在日志里。串口按 8N1 打开，波特率默认 9600。

## 高度

| `sensor_device` | 行为 |
| --- | --- |
| 空 | 使用常量 `sensor_height_m`，视为有效 |
| 普通文件 | 每次读一个浮点数，单位米。可以用 `echo 12.5 > /tmp/height` 注入 |
| `/dev/tty*` | 打开一次，非阻塞读。协议是一行一个浮点数 |

读失败时沿用 `sensor_height_m`，这一次样本标记为无效，变焦增益保持 1。

## 停机

`request_stop()` 只写 `std::atomic<bool>`，给信号处理器用。真正的 join 在主线程的 `wait()` / `stop()`。

拉流线程在循环条件里看这个标志，FFmpeg 的中断回调也看它。推理线程不因为标志立刻扔掉队列里已经解出来的帧；它一直取到队列被关闭并且取空。显示线程在帧队列关闭后退出，退出前析构会放开还握着的两张 NV12。业务线程在 50 ms 超时返回后看到标志就退出。

文件正常结束时，拉流线程自己把标志置上，主线程就会走同一套 `stop()`。
