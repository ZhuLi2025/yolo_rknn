# 模型

运行时只加载已经转好的 `.rknn` 文件，工程里不做模型转换。

本机 NPU 用户态是 RKNN Runtime 2（`librknnrt.so` 1.5.2），对应 RKNN-Toolkit2，目标平台填 `rk3588`。
用 Toolkit 1.x 转给 RK3399Pro 的模型，以及转给 RV1126 / RK3566 的模型，都不能在这里加载。

输入尺寸跟模型走，写在 `model_width` / `model_height`，并且必须是 16 的倍数。YOLOv8 常见 640，YOLOv3-tiny 常见 416。uint8 RGB。类型和布局已经是 uint8 NHWC、宽高没有额外 stride 时，RGA 的 dma-buf 以 `pass_through` 交给 NPU。否则 `rknn_set_io_mem` 仍从这块 fd 读入，由驱动做布局转换。fd 导入失败时才退回 `rknn_inputs_set`。

`npu_cores` 默认 `0_1_2`，一帧使用 RK3588 的三颗 NPU。`model_type=yolov5`、`yolov8` 或 `yolov3` 切换后处理，说明见 [模型与后处理](../docs/05-model.md)。

当前板上的 `weights/yolov5_rk3588_int8.rknn` 是 640、单类、三个 DFL 尺度。配置里 `num_classes=1`，`dfl_scores_are_logits=false`。

把文件放到这个目录，再在 `config/app.conf` 里改 `model_type` 和 `model_path`。
`.rknn` 不进版本库。
