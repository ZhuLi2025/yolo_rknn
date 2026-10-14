/**
 * @file yolov8_detector.hpp
 * @brief RKNN 1.7.5 推理封装。只在推理线程里创建和调用。
 * model_type 决定后处理走 YOLOv8 还是 YOLOv3，加载和 rknn_run 是同一条路径。
 *
 * 本机 API 没有 “从 dma-buf fd 创建输入内存” 的接口，rknn_inputs_set 会把
 * 模型输入张量拷进片外 NPU。调用方应保证 rgb 已经是 RGA 产好的紧凑 RGB888。
 */
#pragma once

#include "yolo_rknn/common/app_config.hpp"
#include "yolo_rknn/common/status.hpp"
#include "yolo_rknn/infer/detection.hpp"
#include "yolo_rknn/preprocess/rga_preprocessor.hpp"

namespace yolo_rknn {

class YoloV8Detector {
public:
    explicit YoloV8Detector(const AppConfig& config);
    ~YoloV8Detector();

    YoloV8Detector(const YoloV8Detector&) = delete;
    YoloV8Detector& operator=(const YoloV8Detector&) = delete;

    Status init();
    Status infer(const PreprocessResult& input, InferenceResult& result);

private:
    struct Impl;
    AppConfig config_;
    Impl* impl_ = nullptr;
};

}  // namespace yolo_rknn
