/**
 * @file postprocess.hpp
 * @brief YOLOv8 输出解码与 NMS。只处理模型张量，不接触图像像素。
 *
 * 支持两种 RKNN 导出形态：
 *  1. 单输出锚点头：[1, 4+nc, N] 或 [1, N, 4+nc]，框是模型输入坐标系下的 cx,cy,w,h。
 *     Ultralytics 导出再转 RKNN 时最常见，类别分数通常已经过 sigmoid。
 *  2. 三个 DFL 检测头：每个是 NCHW/NHWC 的特征图，通道 = 64 + nc（reg_max=16）。
 *     stride 用 model_width / grid_width 推算，不把 8/16/32 写死。
 */
#pragma once

#include "yolo_rknn/infer/detection.hpp"

#include <cstddef>
#include <vector>

namespace yolo_rknn {

struct LetterboxMeta {
    float scale = 1.f;     // 原图边长乘上 scale 得到放进模型输入的边长
    int pad_x = 0;
    int pad_y = 0;
    int src_width = 0;
    int src_height = 0;
    int model_width = 0;
    int model_height = 0;
};

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
    LetterboxMeta letterbox;
};

struct DecodeResult {
    bool layout_recognized = false;
    std::vector<Detection> detections;
};

DecodeResult decode_yolov8(const std::vector<RawTensor>& tensors, const DecodeParams& params);

}  // namespace yolo_rknn
