/**
 * @file postprocess.hpp
 * @brief YOLO 输出解码与 NMS。只处理模型张量，不接触图像像素。
 *
 * model_type=yolov8 走 decode_yolov8，两种 RKNN 导出形态：
 *  1. 单输出锚点头：[1, 4+nc, N] 或 [1, N, 4+nc]，框是模型输入坐标系下的 cx,cy,w,h。
 *     Ultralytics 导出再转 RKNN 时最常见，类别分数通常已经过 sigmoid。
 *  2. 三个 DFL 检测头：每个是 NCHW/NHWC 的特征图，通道 = 64 + nc（reg_max=16）。
 *     stride 用 model_width / grid_width 推算，不把 8/16/32 写死。
 *
 * model_type=yolov3 走 decode_yolov3。darknet 的 YOLOv3 / YOLOv3-tiny：
 * 每个头通道数是 3 * (5 + nc)，框用 sigmoid 加网格、exp 乘 anchor。
 * Toolkit 1.x 常把 NCHW 的维度倒着填成 [W, H, C, N]，两种填法都认。
 *
 * model_type=yolov5 走 decode_yolov5，两种形态：
 *  1. 本机 weights/yolov5_rk3588_int8.rknn：三个尺度，每个尺度一张 64 通道 DFL 框，
 *     再加若干张 nc 通道分数图。同一尺度取输出顺序里最后一张分数图。
 *  2. 经典 YOLOv5 锚点头：3 * (5 + nc)。框是 (sigmoid*2-0.5+grid)*stride，
 *     宽高是 (sigmoid*2)^2 * anchor。
 */
#pragma once

#include "yolo_rknn/infer/detection.hpp"

#include <cstddef>
#include <vector>

namespace yolo_rknn {

struct LetterboxMeta {
    float scale = 1.f;  // 原图边长乘上 scale 得到放进模型输入的边长
    // letterbox 时是灰边。short_side 时是负的，表示模型坐标 0 左边被裁掉的原图距离（已乘 scale）。
    float pad_x = 0.f;
    float pad_y = 0.f;
    int src_width = 0;
    int src_height = 0;
    int model_width = 0;
    int model_height = 0;
};

// letterbox：整图缩进模型，空边填灰。short_side：短边缩放到模型边，长边裁掉。
enum class InputFit {
    Letterbox = 0,
    ShortSide = 1,
};

// short_side 的裁剪锚点。Center 是单张居中。Start/End 贴长边的两端，给 batch=2 的左右（或上下）两张。
enum class WindowAnchor {
    Center = 0,
    Start = 1,
    End = 2,
};

// RGA 实际采样的原图矩形，以及它写进模型输入的矩形。坐标都是偶数对齐后的像素。
struct ModelWindow {
    int src_x = 0;
    int src_y = 0;
    int src_w = 0;
    int src_h = 0;
    int dst_x = 0;
    int dst_y = 0;
    int dst_w = 0;
    int dst_h = 0;
    bool fill = true;
    LetterboxMeta meta;
};

bool plan_model_window(InputFit fit, int src_w, int src_h, int model_w, int model_h, ModelWindow* window,
                       WindowAnchor anchor = WindowAnchor::Center);

// 两张切片映回原图后去重。除了 IoU，小框大部分落在大框里也算同一个目标。
std::vector<Detection> suppress_overlaps(std::vector<Detection> dets, float iou_threshold, int max_det);

struct RawTensor {
    std::vector<int> dims;
    bool channel_last = false;  // true 表示 NHWC，false 表示 NCHW
    const float* data = nullptr;
    std::size_t n_elems = 0;
};

struct DecodeParams {
    int num_classes = 80;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int max_detections = 64;
    bool anchor_scores_are_probabilities = true;
    bool dfl_scores_are_logits = true;
    // yolov3：宽、高交替，3 个 anchor 为一组，组的顺序是 stride 从大到小。
    std::vector<float> anchors;
    LetterboxMeta letterbox;
};

struct DecodeResult {
    bool layout_recognized = false;
    std::vector<Detection> detections;
};

DecodeResult decode_yolov8(const std::vector<RawTensor>& tensors, const DecodeParams& params);
DecodeResult decode_yolov3(const std::vector<RawTensor>& tensors, const DecodeParams& params);
DecodeResult decode_yolov5(const std::vector<RawTensor>& tensors, const DecodeParams& params);

// 不看数值，只看 shape 能不能当成对应检测头。初始化时用来尽早失败。
bool yolov3_outputs_recognized(const std::vector<RawTensor>& tensors, int num_classes, int model_width,
                               int model_height);
bool yolov5_split_dfl_recognized(const std::vector<RawTensor>& tensors, int num_classes, int model_width,
                                 int model_height);
bool yolov5_outputs_recognized(const std::vector<RawTensor>& tensors, int num_classes, int model_width,
                               int model_height);

}  // namespace yolo_rknn
