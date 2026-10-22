# 模型与后处理

## 转换

运行时加载 `.rknn`，不在板子上做转换。本机库是 RKNN Runtime 2（`librknnrt.so` 1.5.2），模型要用 RKNN-Toolkit2 导出，`target_platform` 填 `rk3588`。

Toolkit 1.x 的产物（RK3399Pro）和其他平台的 Toolkit2 模型都不兼容，`rknn_init` 会失败。失败日志里会写明这一点。

输入请做成：

- 640x640（或和 `model_width` / `model_height` 一致，且为 16 的倍数）
- uint8
- NHWC
- RGB，数值范围 0–255，不要在模型里再要求减均值。RGA 输出的就是这张图

尺寸和 `n_elems` 对不上时，初始化直接失败，并打印 `rknn_query` 看到的 shape。对得上、并且类型已经是 uint8 NHWC、宽高 stride 没有额外填充时，`pass_through=1`，驱动不再做一次布局转换。类型是 float、NCHW，或带 stride 时，`pass_through=0`，由驱动把这块 dma-buf 里的 RGB888 转成模型内部输入。

`rknn_set_core_mask` 按 `npu_cores` 设置。默认 `0_1_2`，三核一起跑这一帧。`RKNN_FLAG_ASYNC_MASK` 没有打开。那个模式取回的是上一帧的结果，业务变焦会跟着旧框走。

输出缓冲在初始化时按 `n_elems * sizeof(float)` 预分配，`rknn_outputs_get` 使用 `want_float=1` 和 `is_prealloc=1`。后处理因此不用每帧 malloc，量化反算留在驱动里。

## 用配置切换版本

`model_type` 决定后处理，加载和 `rknn_run` 不分开。

| `model_type` | 典型输入 | 输出 |
| --- | --- | --- |
| `yolov5` | 640x640 | 三个尺度的 64 通道 DFL 框加 nc 通道分数，或三个 `3*(5+nc)` 锚点头 |
| `yolov8` | 640x640 | 一张 `[1, 4+nc, N]`，或三个合并的 DFL 头 |
| `yolov3` | 416x416 | YOLOv3-tiny 两个头，或完整 YOLOv3 三个头 |

`model_width` / `model_height` 必须和 `rknn_query` 看到的输入一致。对不上时初始化失败，日志里同时有配置尺寸和张量 shape。

换模型时改 `config/app.conf` 里的 `model_type`、`model_path`、`model_width`、`model_height`。锚点头还要看 `anchors`。

当前默认模型是 `weights/yolov5_rk3588_int8.rknn`。输入是 uint8 NHWC 640x640，内部量化成 int8，`scale=1/255`、`zp=-128`，所以 RGA 的 RGB888 用 `pass_through=0` 送进去。输出是 9 张 NCHW 图，三个尺度各一组：

| 尺度 | 框 | 分数 |
| --- | --- | --- |
| 80x80，stride 8 | `[1, 64, 80, 80]` | 两张 `[1, 1, 80, 80]`，用后面那张 |
| 40x40，stride 16 | `[1, 64, 40, 40]` | 两张 `[1, 1, 40, 40]`，用后面那张 |
| 20x20，stride 32 | `[1, 64, 20, 20]` | 两张 `[1, 1, 20, 20]`，用后面那张 |

64 通道是 4 边 × 16 个 DFL bin。分数已经在 0–1，`dfl_scores_are_logits=false`，`num_classes=1`。前面那张同尺寸分数是导出时留下的 Sigmoid，解码不用它。

## YOLOv8 的两种输出布局

`decode_yolov8` 只认下面两种。其它 shape 会把 `layout_recognized` 设为 false，推理线程打一条错误日志并丢掉该帧。这样比静默出 0 个框更容易发现模型导错了。

### 单张锚点头

Ultralytics 导出再转 RKNN 时最常见。

- shape `[1, 4+nc, N]` 或 `[1, N, 4+nc]`，也接受没有 batch 维的二维形式
- 前 4 个数是模型输入坐标系下的 `cx, cy, w, h`，单位是像素，不是 0–1 的归一化坐标
- 后面 `nc` 个是类别分数
- `anchor_scores_are_probabilities=true` 时不再做 sigmoid。如果导出的还是 logits，把这项改成 false

`nc` 来自配置 `num_classes`，默认 80。通道数对不上就认为不是这种布局。

### 三个 DFL 头

每个输出是 4 维特征图，通道数 `64 + nc`（`reg_max = 16`）。

- NCHW：`[1, C, H, W]`
- NHWC：`[1, H, W, C]`，由张量的 `fmt` 决定怎么索引
- `stride = model_width / W`。W 不能整除模型宽的头会被跳过
- 前 64 通道是 4 边 × 16 个 bin 的分布，做 softmax 期望得到 l,t,r,b
- 框：`(grid + 0.5 ± dist) * stride`
- `dfl_scores_are_logits=true` 时对类别通道做 sigmoid

三个头都解析失败才算布局不识别。某个头的类别分数极低时可以没有框，这是正常的空场景。

## YOLOv3

`decode_yolov3` 按 darknet 的检测头解码。YOLOv3-tiny 是 2 个头，完整 YOLOv3 是 3 个头。每个头的通道数是 `3 * (5 + nc)`：3 个 anchor，每个 anchor 是 `tx, ty, tw, th, objectness` 再加 `nc` 个类别。

Toolkit2 在 RK3588 上的维度顺序以 `rknn_query` 的日志为准。下面这组是以前在 RK3399Pro、Toolkit 1.x 上看到的，换平台后不要照抄，只说明后处理同时认这几种排法：

- 输入常见为 uint8，416x416
- 输出可能是 `[1, 255, 13, 13]`，也可能被 1.x 记成 `[13, 13, 255, 1]`
- `fmt` 决定按 NCHW 还是 NHWC 索引

标准的 `[1, C, H, W]` 和 NHWC 的 `[1, H, W, C]` 都认。stride 用 `model_width / grid_width` 算。

解码：

```text
bx = (sigmoid(tx) + grid_x) * stride
by = (sigmoid(ty) + grid_y) * stride
bw = exp(tw) * anchor_w
bh = exp(th) * anchor_h
score = sigmoid(objectness) * sigmoid(class)
```

`anchors` 是模型输入像素下的宽高，按 stride 从大到小，每个 stride 三个。空着不写时，2 个头用 YOLOv3-tiny 的 COCO anchor，3 个头用完整 YOLOv3 的 9 个。个数和输出头对不上，初始化失败。

## 坐标回到原图

RGA letterbox 的 `scale`、`pad_x`、`pad_y` 放在 `LetterboxMeta` 里。

```text
原图坐标 = (模型坐标 - pad) / scale
```

然后夹到原图范围内。宽或高小于 1 像素的框丢掉。

NMS 按类别分开，IoU 阈值是 `nms_threshold`，保留不超过 `max_detections` 个，按分数从高到低。

## 业务怎么用框

业务线程在一帧的全部框里，先按 `target_class` 过滤（负值表示不过滤），再取高度最大的一个框，用它的高度除以画面高度作为占比。类别名使用 COCO 80 类，只用于日志。

## 和测试的关系

`tests/test_core.cpp` 构造了一张 `[1, 84, 2]` 的锚点头，两个重叠框，确认 NMS 只留高分的那个，左上角大约在 (270, 280)，尺寸 100x80。

DFL 测试用 16x16 的模型输入、stride 8 的 2x2 特征图，bin 1 为独热，期望框夹紧到图像内后是 (0, 0, 12, 12)。另外两个头的类别 logit 设成很大的负数，不应该出框。

这两项不过，就不要把后处理的问题当成 NPU 或 RGA 的问题。
