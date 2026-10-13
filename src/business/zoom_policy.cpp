#include "yolo_rknn/business/zoom_policy.hpp"

namespace yolo_rknn {

ZoomAction decide_zoom(const ZoomInput& input) {
    // 没有目标就不动。用高度去空转镜头，停不下来。
    if (!input.has_target || input.target_ratio <= 0.f) {
        return ZoomAction::Hold;
    }

    // 正误差：框比目标小，需要放大。
    float error = input.target_ratio - input.bbox_h_ratio;
    float gain = 1.f;
    if (input.height_valid && input.height_m > 0.05f && input.reference_height_m > 0.05f) {
        gain = input.height_m / input.reference_height_m;
        if (gain < 0.5f) {
            gain = 0.5f;
        } else if (gain > 2.f) {
            gain = 2.f;
        }
    }

    // 增益只改变“多快越过死区”，误差为 0 时结果仍是保持。
    const float score = error * gain;
    const float band = input.target_ratio * input.deadband;
    if (score > band) {
        return ZoomAction::ZoomIn;
    }
    if (score < -band) {
        return ZoomAction::ZoomOut;
    }
    return ZoomAction::Hold;
}

}  // namespace yolo_rknn
