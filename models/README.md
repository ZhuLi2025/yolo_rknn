# 模型

运行时只加载已经转好的 `.rknn` 文件，工程里不做模型转换。

本机 NPU 用户态是 RKNN API 1.7.5，对应 RKNN-Toolkit 1.x，目标平台填 `rk3399pro`。
用 RKNN-Toolkit2 转给 RV1126 / RK3588 的模型不能在这里加载。

建议的输入：

- 640x640，uint8，NHWC，RGB
- 这样 RGA 产出来的缓冲可以直接 `pass_through` 送进 NPU，驱动少做一次布局转换

支持的输出见 [模型与后处理](../docs/05-model.md)。

把文件放到这个目录，例如 `models/yolov8n.rknn`，再在 `config/app.conf` 里改 `model_path`。
`.rknn` 不进版本库。
