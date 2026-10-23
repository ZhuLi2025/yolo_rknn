/**
 * @file detection.hpp
 * @brief 检测框使用原图像素坐标，原点在左上角，x/y 是左上角而不是中心。
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace yolo_rknn {

struct Detection {
    float x = 0.f;
    float y = 0.f;
    float w = 0.f;
    float h = 0.f;
    float score = 0.f;
    int class_id = -1;
    // -1 未做头盔分类。0 有头盔，1 无头盔。不覆盖 class_id，变焦过滤仍看检测类别。
    int helmet = -1;
    float helmet_score = 0.f;
};

struct InferenceResult {
    std::uint64_t seq = 0;
    std::int64_t pts_us = -1;
    int src_width = 0;
    int src_height = 0;
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    std::vector<Detection> detections;
};

// COCO 80 类。类别号超出范围时返回 "unknown"。
const char* coco_name(int class_id);

}  // namespace yolo_rknn
