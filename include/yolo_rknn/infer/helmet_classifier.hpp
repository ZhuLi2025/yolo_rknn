/**
 * @file helmet_classifier.hpp
 * @brief 对每个检测框做头盔二分类。输入是原图 NV12 上的 ROI，不是检测模型的 640 输入。
 *
 * 模型 weights/cls_rk3588_int8.rknn：128x128 RGB，输出 [1,2]。
 * 通道 0 是有头盔，通道 1 是无头盔。跑在 NPU core2，避免和左右两张检测抢核。
 */
#pragma once

#include "yolo_rknn/common/app_config.hpp"
#include "yolo_rknn/common/status.hpp"
#include "yolo_rknn/decode/decoded_frame.hpp"
#include "yolo_rknn/infer/detection.hpp"

#include <vector>

namespace yolo_rknn {

class HelmetClassifier {
public:
    explicit HelmetClassifier(const AppConfig& config);
    ~HelmetClassifier();

    HelmetClassifier(const HelmetClassifier&) = delete;
    HelmetClassifier& operator=(const HelmetClassifier&) = delete;

    Status init();
    // frame 的 NV12 fd 必须仍然有效。成功时写入 helmet 和 helmet_score。
    Status classify(const DecodedFrame& frame, std::vector<Detection>& detections);

private:
    struct Impl;
    AppConfig config_;
    Impl* impl_ = nullptr;
};

}  // namespace yolo_rknn
