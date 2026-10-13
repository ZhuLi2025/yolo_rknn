/**
 * @file controller.hpp
 * @brief 业务线程：读高度、按检测框计算变焦、向相机发 VISCA。
 *
 * 只在业务线程调用。检测结果由 pipeline 从队列取出后送进来。
 */
#pragma once

#include "yolo_rknn/business/zoom_policy.hpp"
#include "yolo_rknn/common/app_config.hpp"
#include "yolo_rknn/infer/detection.hpp"

#include <cstdint>

namespace yolo_rknn {

class Controller {
public:
    explicit Controller(const AppConfig& config);
    ~Controller();

    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;

    void on_inference(const InferenceResult& result);
    void poll();  // 没有新检测时也周期调用，用来读传感器、补发停止指令

private:
    struct Impl;
    void apply(ZoomAction action);
    float read_height(bool* valid);

    AppConfig config_;
    Impl* impl_ = nullptr;
    InferenceResult latest_;
    bool has_result_ = false;
    ZoomAction last_action_ = ZoomAction::Hold;
    std::int64_t last_cmd_ms_ = 0;
};

}  // namespace yolo_rknn
