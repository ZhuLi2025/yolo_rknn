/**
 * @file yolov8_detector.hpp
 * @brief RK3588 上的 RKNN Runtime 2 推理封装。只在推理线程里创建和调用。
 * model_type 决定后处理走 YOLOv8 还是 YOLOv3，加载和 rknn_run 是同一条路径。
 *
 * bind_input() 用 rknn_create_mem_from_fd 把 RGA 的目标 dma-buf 交给片内 NPU。
 * 导入失败时 infer() 退回 rknn_inputs_set，仍然要求 rgb 是紧凑 RGB888。
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
    // fd 在整个推理线程的生命周期里保持有效。失败只记日志，不让 pipeline 退出。
    void bind_input(int fd, void* virt, std::size_t bytes);
    Status infer(const PreprocessResult& input, InferenceResult& result);

private:
    struct Impl;
    AppConfig config_;
    Impl* impl_ = nullptr;
};

}  // namespace yolo_rknn
