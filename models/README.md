# 模型

运行时只加载已经转好的 `.rknn` 文件，工程里不做模型转换。

本机 NPU 用户态是 RKNN API 1.7.5，对应 RKNN-Toolkit 1.x，目标平台填 `rk3399pro`。
用 RKNN-Toolkit2 转给 RV1126 / RK3588 的模型不能在这里加载。

输入尺寸跟模型走，写在 `model_width` / `model_height`。YOLOv8 常见 640，YOLOv3-tiny 常见 416。uint8 RGB。类型和布局已经是 uint8 NHWC 时，RGA 的缓冲可以 `pass_through` 送进 NPU。

`model_type=yolov8` 或 `yolov3` 切换后处理，说明见 [模型与后处理](../docs/05-model.md)。

把文件放到这个目录，再在 `config/app.conf` 里改 `model_type` 和 `model_path`。
`.rknn` 不进版本库。
