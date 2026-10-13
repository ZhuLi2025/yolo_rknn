/**
 * @file rtsp_client.hpp
 * @brief FFmpeg 只负责解封装。这里没有 avcodec_open2，也没有把 AVFrame 解成像素。
 *
 * 压缩包从解复用器出来后，经 h264_mp4toannexb / hevc_mp4toannexb 转成 Annex-B，
 * 再交给 MPP。码流拷贝是几十字节到几十 KB，和整帧 NV12 不是一个量级。
 */
#pragma once

#include "yolo_rknn/common/app_config.hpp"
#include "yolo_rknn/common/status.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace yolo_rknn {

struct EncodedPacket {
    std::vector<std::uint8_t> data;
    std::int64_t pts_us = -1;
    bool keyframe = false;
};

enum class ReadStatus {
    Packet,   // out 里有一包 Annex-B
    Eos,      // 文件结束。直播流不应把它当成永久失败
    Stopped,  // 外部把停止标志置上，中断回调打断了阻塞读
    Error     // 网络或解复用错误，调用方可以重连
};

class RtspClient {
public:
    RtspClient(const AppConfig& config, const std::atomic<bool>* stop);
    ~RtspClient();

    RtspClient(const RtspClient&) = delete;
    RtspClient& operator=(const RtspClient&) = delete;

    Status open();
    Status reopen();
    ReadStatus read(EncodedPacket& out);

    int video_width() const { return width_; }
    int video_height() const { return height_; }
    const std::string& codec_name() const { return codec_name_; }

private:
    struct FfmpegState;

    static int interrupt_cb(void* opaque);
    Status open_input();
    void close_input();

    AppConfig config_;
    const std::atomic<bool>* stop_ = nullptr;
    FfmpegState* ff_ = nullptr;
    std::deque<EncodedPacket> pending_;
    int width_ = 0;
    int height_ = 0;
    std::string codec_name_;
};

}  // namespace yolo_rknn
