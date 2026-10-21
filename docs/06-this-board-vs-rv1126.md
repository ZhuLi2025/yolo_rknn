# 这台板子

需求是按 RV1126 的习惯写的：MPI、MMZ、片内 NPU 零拷贝。当前内核设备树是：

```text
SuperString RK3588 RB1005 Board
superstring,rk3588-rb1005
rockchip,rk3588
```

CPU 是 Cortex-A76 ×4 + Cortex-A55 ×4。NPU 是三核 RKNN2，节点在 `fdab0000.npu`，对应 `/dev/dri/card1` 和 `/dev/dri/renderD129`。工程按这块板子能链接、能打开的设备来写。

## 对照

| 需求里的说法 | 本机实际 | 代码位置 |
| --- | --- | --- |
| 减小拉流缓冲，`ff_url_set_option` 的 `max_delay`、`buffer_size` | 用公开 API `av_dict_set` 写同名 AVOption，并设置 `AVFMT_FLAG_NOBUFFER` | `src/stream/rtsp_client.cpp` |
| 传输用 TCP | 默认 `transport=tcp`，UDP 必须 `allow_udp=true` 才放行 | `AppConfig::validate` |
| 禁止 CPU 软解，走 MPI / VPU，输出 NV12 | Rockchip MPP 的 `rk_mpi.h`。解码类型只接受 H.264/H.265，输出强制线性 NV12。AFBC 帧直接丢掉 | `src/decode/mpp_decoder.cpp` |
| 图像放在 MMZ | 优先 `MPP_BUFFER_TYPE_DRM \| CONTIG`。当前用户打得开 `/dev/dri`，打不开 `/dev/dma_heap`，所以 DRM CMA 在前，dma-heap 只做后备 | `MppDecoder::ensure_buffer_group` |
| 预处理交给 RGA，CPU 不搬大图 | librga 的 im2d：`imfill` + `improcess`。驱动里有 RGA2 和 RGA3，由库选核。工程不链接 `libswscale` | `src/preprocess/rga_preprocessor.cpp` |
| NPU 零拷贝 | `rknn_create_mem_from_fd` + `rknn_set_io_mem` 把 RGA 的目标 fd 交给片内 NPU。导入失败才退回 `rknn_inputs_set` | `src/infer/yolov8_detector.cpp` |
| 三条线程，拉流硬解高优先级，推理和业务分开 | `yr_stream` / `yr_infer` / `yr_biz`，绑核加尝试 `SCHED_FIFO` | `src/pipeline/pipeline.cpp` |

## 和 RK3399Pro 不一样的地方

这份代码原先按 RK3399Pro 写。换到 RK3588 时改了下面几处，不要再按那块板子的假设改回去：

1. 用户态库是 `/usr/lib/librknnrt.so`，头文件是 `/usr/include/rknn_api.h`。`rknn_init` 有第五个 `extend` 参数。没有 `librknn_api.so`。
2. NPU 在片内，三核。`npu_cores=0_1_2` 调用 `rknn_set_core_mask`。这个接口在 RK3588 上才有意义。
3. 模型必须用 RKNN-Toolkit2 重新转换，`target_platform=rk3588`。rk3399pro 的 `.rknn` 在这里 `rknn_init` 会失败。
4. 绑核按 A76 / A55。cpu0–3 是 A55，cpu4–5 和 cpu6–7 是两组 A76。不要再把 cpu4 当成 A72。
5. RGA 走 im2d。模型宽高必须是 16 的倍数，RGA3 才接受 RGB888。
6. 显示控制器在 `/dev/dri/card0`。`card1` 是 NPU 的 DRM 节点，`display_device` 不能写成它。

拉流侧的 TCP、小缓冲、Annex-B、三条线程和“丢旧帧”没有因换芯片改掉。那些限制来自实时视频。
