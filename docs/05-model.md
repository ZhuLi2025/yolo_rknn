# 模型与后处理

## 转换

运行时加载 `.rknn`，不在板子上做转换。本机库是 RKNN API 1.7.5，模型要用 RKNN-Toolkit 1.x 导出，`target_platform` 填 `rk3399pro`。

Toolkit2 的产物（RV1126、RK3566、RK3588）和这份 API 不兼容，`rknn_init` 会失败。失败日志里会写明这一点。

输入请做成：

- 640x640（或和 `model_width` / `model_height` 一致，且为偶数）
- uint8
- NHWC
- RGB，数值范围 0–255，不要在模型里再要求减均值。RGA 输出的就是这张图

尺寸和 `n_elems` 对不上时，初始化直接失败，并打印 `rknn_query` 看到的 shape。对得上、并且类型已经是 uint8 NHWC 时，`pass_through=1`，驱动不再做一次布局转换。类型是 float 或 NCHW 时，`pass_through=0`，由驱动把这张 RGB888 转成模型内部输入。这一步仍然只发生在模型输入上。

`RKNN_FLAG_ASYNC_MASK` 没有打开。那个模式取回的是上一帧的结果，业务变焦会跟着旧框走。

输出缓冲在初始化时按 `n_elems * sizeof(float)` 预分配，`rknn_outputs_get` 使用 `want_float=1` 和 `is_prealloc=1`。后处理因此不用每帧 malloc，量化反算留在驱动里。

## 用配置切换版本

`model_type` 决定后处理，加载和 `rknn_run` 不分开。

| `model_type` | 典型输入 | 输出 |
| --- | --- | --- |
| `yolov8` | 640x640 | 一张 `[1, 4+nc, N]`，或三个 DFL 头 |
| `yolov3` | 416x416 | YOLOv3-tiny 两个头，或完整 YOLOv3 三个头 |

`model_width` / `model_height` 必须和 `rknn_query` 看到的输入一致。对不上时初始化失败，日志里同时有配置尺寸和张量 shape。

换模型时改 `config/app.conf` 里的 `model_type`、`model_path`、`model_width`、`model_height`。YOLOv3 还要看 `anchors`。

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

`decode_yolov3` 按 darknet 的检测头解码。YOLOv3-tiny 是 2 个头，完整 YOLOv3 是 3 个头。每个头的通道数是 `3 * (5 + nc)`：3 个 anchor，每个 anchor 是 `tx, ty, tw, th, objectness` 再加 `nc` 个类别。本机的 `yolov3_tiny_rk3399pro.rknn` 查询结果是：

- 输入 `[416, 416, 3, 1]`，uint8，416x416
- 输出 `[13, 13, 255, 1]` 和 `[26, 26, 255, 1]`，`fmt` 是 NCHW

Toolkit 1.x 把 NCHW 的维度倒着填，所以 `[13, 13, 255, 1]` 的内存仍是 `[1, 255, 13, 13]` 的平面排列。标准的 `[1, C, H, W]` 和 NHWC 的 `[1, H, W, C]` 也认。stride 用 `model_width / grid_width` 算，416 上分别是 32 和 16。

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
