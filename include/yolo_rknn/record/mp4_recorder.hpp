/**
 * @file mp4_recorder.hpp
 * @brief 把展示画面录成 H.264 MP4。画面是原分辨率 NV12 叠上和 HDMI OSD 相同的框和标签。
 *
 * 编码走 VPU，像素转换走 RGA。只在录制线程里调用。
 */
#pragma once

#include "yolo_rknn/common/app_config.hpp"
#include "yolo_rknn/common/status.hpp"
#include "yolo_rknn/decode/decoded_frame.hpp"
#include "yolo_rknn/infer/detection.hpp"

#include <cstdint>
#include <string>

namespace yolo_rknn {

struct RecordSample {
    DecodedFrame frame;
    InferenceResult result;
};

class Mp4Recorder {
public:
    explicit Mp4Recorder(const AppConfig& config);
    ~Mp4Recorder();

    Mp4Recorder(const Mp4Recorder&) = delete;
    Mp4Recorder& operator=(const Mp4Recorder&) = delete;

    // 创建目录并定下文件名。编码器要等第一帧才知道分辨率。
    Status prepare();
    Status write(const RecordSample& sample);
    void close();

    const std::string& path() const;
    std::uint64_t frames() const;

private:
    Status write_packet(void* packet, std::int64_t pts_us);
    struct Impl;
    AppConfig config_;
    Impl* impl_ = nullptr;
};

}  // namespace yolo_rknn
