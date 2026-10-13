/**
 * @file zoom_policy.hpp
 * @brief 变焦闭环。有检测框时以画面占比为误差，高度只放大误差的增益，不单独把镜头推到头。
 *
 * 这样写是为了能停住：占比已经达到目标时误差为 0，飞得再高也不会继续放大。
 * 没有目标时保持不动，避免镜头空转。
 */
#pragma once

namespace yolo_rknn {

enum class ZoomAction {
    Hold,
    ZoomIn,
    ZoomOut
};

struct ZoomInput {
    bool has_target = false;
    float bbox_h_ratio = 0.f;         // 框高 / 画面高
    bool height_valid = false;
    float height_m = 0.f;
    float reference_height_m = 10.f;
    float target_ratio = 0.40f;
    float deadband = 0.20f;           // 相对目标比例的死区
};

ZoomAction decide_zoom(const ZoomInput& input);

}  // namespace yolo_rknn
