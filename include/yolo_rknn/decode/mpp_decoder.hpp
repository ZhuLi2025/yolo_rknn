/**
 * @file mpp_decoder.hpp
 * @brief 把 Annex-B 码流送进 Rockchip VPU，取出 NV12 dma-buf。
 *
 * 这是整条链路里唯一的解码入口。不要在别处再开一个 FFmpeg 软解。
 * 解码器对象只能由拉流线程调用；MPP context 不是线程安全的。
 */
#pragma once

#include "yolo_rknn/common/app_config.hpp"
#include "yolo_rknn/common/status.hpp"
#include "yolo_rknn/decode/decoded_frame.hpp"
#include "yolo_rknn/stream/rtsp_client.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace yolo_rknn {

enum class VideoCodec {
    H264,
    H265
};

class MppDecoder {
public:
    explicit MppDecoder(const AppConfig& config);
    ~MppDecoder();

    MppDecoder(const MppDecoder&) = delete;
    MppDecoder& operator=(const MppDecoder&) = delete;

    Status init(VideoCodec codec);
    // 可能一次吐出 0 帧或多帧（信息变更帧会被内部消化）。
    Status decode(const EncodedPacket& packet, std::vector<DecodedFrame>& frames);
    // 码流结束时送一个 EOS 空包，把还留在 VPU 里的帧取完。
    Status flush(std::vector<DecodedFrame>& frames);

private:
    struct Impl;
    Status ensure_buffer_group(void* frame);
    Status drain(std::vector<DecodedFrame>& frames);

    AppConfig config_;
    Impl* impl_ = nullptr;
    std::uint64_t seq_ = 0;
    bool warned_format_ = false;
    bool eos_seen_ = false;
};

const char* video_codec_name(VideoCodec codec);

}  // namespace yolo_rknn
