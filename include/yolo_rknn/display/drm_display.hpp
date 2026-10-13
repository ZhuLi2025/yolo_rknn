/**
 * @file drm_display.hpp
 * @brief HDMI 两层显示。视频层扫描 VPU 的 NV12 dma-buf，OSD 层是一块 ARGB。
 *
 * 只在显示线程里调用。NV12 的引用会留到再往下翻两帧，避免显示控制器还在扫这块内存时
 * 解码器把它拿去写下一帧。
 */
#pragma once

#include "yolo_rknn/common/app_config.hpp"
#include "yolo_rknn/common/status.hpp"
#include "yolo_rknn/decode/decoded_frame.hpp"
#include "yolo_rknn/infer/detection.hpp"

namespace yolo_rknn {

class DrmDisplay {
public:
    explicit DrmDisplay(const AppConfig& config);
    ~DrmDisplay();

    DrmDisplay(const DrmDisplay&) = delete;
    DrmDisplay& operator=(const DrmDisplay&) = delete;

    Status init();
    Status present(const DecodedFrame& frame);
    void update_osd(const InferenceResult& result);

private:
    // 本机 RGB overlay 没有缩放器，源尺寸必须等于 CRTC 上的目标矩形。
    bool place_osd_plane();
    struct Impl;
    AppConfig config_;
    Impl* impl_ = nullptr;
};

}  // namespace yolo_rknn
