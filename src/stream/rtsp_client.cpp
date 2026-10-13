#include "yolo_rknn/stream/rtsp_client.hpp"

#include "yolo_rknn/common/logger.hpp"

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
}

#include <cerrno>
#include <cstring>
#include <mutex>

namespace yolo_rknn {
namespace {

std::once_flag g_net_once;

std::string ffmpeg_err(int err) {
    char buf[160];
    av_strerror(err, buf, sizeof(buf));
    return std::string(buf);
}

bool starts_with(const std::string& text, const char* prefix) {
    const std::size_t n = std::strlen(prefix);
    return text.size() >= n && text.compare(0, n, prefix) == 0;
}

}  // namespace

struct RtspClient::FfmpegState {
    AVFormatContext* fmt = nullptr;
    AVBSFContext* bsf = nullptr;
    int video_index = -1;
    AVRational time_base;
};

RtspClient::RtspClient(const AppConfig& config, const std::atomic<bool>* stop)
    : config_(config), stop_(stop), ff_(new FfmpegState) {
    ff_->time_base.num = 0;
    ff_->time_base.den = 1;
}

RtspClient::~RtspClient() {
    close_input();
    delete ff_;
    ff_ = nullptr;
}

int RtspClient::interrupt_cb(void* opaque) {
    RtspClient* self = static_cast<RtspClient*>(opaque);
    if (self && self->stop_ && self->stop_->load()) {
        return 1;
    }
    return 0;
}

Status RtspClient::open() { return open_input(); }

Status RtspClient::reopen() {
    close_input();
    return open_input();
}

void RtspClient::close_input() {
    pending_.clear();
    if (ff_ == nullptr) {
        return;
    }
    if (ff_->bsf) {
        av_bsf_free(&ff_->bsf);
        ff_->bsf = nullptr;
    }
    if (ff_->fmt) {
        avformat_close_input(&ff_->fmt);
        ff_->fmt = nullptr;
    }
    ff_->video_index = -1;
}

Status RtspClient::open_input() {
    std::call_once(g_net_once, [] { avformat_network_init(); });
    close_input();

    AVFormatContext* fmt = avformat_alloc_context();
    if (!fmt) {
        return Status::Fail("avformat_alloc_context failed");
    }
    fmt->interrupt_callback.callback = &RtspClient::interrupt_cb;
    fmt->interrupt_callback.opaque = this;
    // nobuffer + 很小的 max_delay：解复用层不要为了平滑抖动去攒包。
    fmt->flags |= AVFMT_FLAG_NOBUFFER;
    fmt->max_delay = config_.max_delay_us;
    fmt->probesize = config_.probe_size > 0 ? config_.probe_size : 32768;
    fmt->max_analyze_duration = config_.analyze_duration_us;

    AVDictionary* opts = nullptr;
    const bool rtsp = starts_with(config_.url, "rtsp://") || starts_with(config_.url, "rtsps://");
    if (rtsp) {
        // 这些键就是协议层的 AVOption，过去在库内部叫 ff_url_set_option。
        // 应用代码不能调用那个私有符号，av_dict_set 会写到同一组选项上。
        av_dict_set(&opts, "rtsp_transport", config_.transport.c_str(), 0);
        av_dict_set(&opts, "buffer_size", std::to_string(config_.buffer_size).c_str(), 0);
        av_dict_set(&opts, "max_delay", std::to_string(config_.max_delay_us).c_str(), 0);
        av_dict_set(&opts, "reorder_queue_size", std::to_string(config_.reorder_queue_size).c_str(), 0);
        av_dict_set(&opts, "stimeout", std::to_string(config_.stimeout_us).c_str(), 0);
        if (config_.transport == "udp") {
            YR_LOGW("RTSP over UDP. Packet loss will stall the hardware decoder");
        }
    }
    av_dict_set(&opts, "fflags", "nobuffer", 0);

    YR_LOGI("open %s transport=%s max_delay_us=%d buffer_size=%d", config_.url.c_str(),
            rtsp ? config_.transport.c_str() : "n/a", config_.max_delay_us, config_.buffer_size);

    const int opened = avformat_open_input(&fmt, config_.url.c_str(), nullptr, &opts);
    av_dict_free(&opts);
    if (opened < 0) {
        // 失败时 FFmpeg 会释放并清空 fmt。
        return Status::Fail("avformat_open_input: " + ffmpeg_err(opened));
    }
    ff_->fmt = fmt;

    const int info = avformat_find_stream_info(ff_->fmt, nullptr);
    if (info < 0) {
        close_input();
        return Status::Fail("avformat_find_stream_info: " + ffmpeg_err(info));
    }

    int video_index = -1;
    for (unsigned i = 0; i < ff_->fmt->nb_streams; ++i) {
        AVStream* stream = ff_->fmt->streams[i];
        if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && video_index < 0) {
            video_index = static_cast<int>(i);
        } else {
            // 音频包不进解码线程，减少解复用后的无用拷贝。
            stream->discard = AVDISCARD_ALL;
        }
    }
    if (video_index < 0) {
        close_input();
        return Status::Fail("no video stream");
    }

    AVStream* stream = ff_->fmt->streams[video_index];
    AVCodecParameters* par = stream->codecpar;
    const char* bsf_name = nullptr;
    if (par->codec_id == AV_CODEC_ID_H264) {
        bsf_name = "h264_mp4toannexb";
        codec_name_ = "h264";
    } else if (par->codec_id == AV_CODEC_ID_HEVC) {
        bsf_name = "hevc_mp4toannexb";
        codec_name_ = "h265";
    } else {
        close_input();
        return Status::Fail("only H.264/H.265 can be sent to the VPU; software decode is not implemented");
    }

    const AVBitStreamFilter* filter = av_bsf_get_by_name(bsf_name);
    if (!filter) {
        close_input();
        return Status::Fail(std::string("bitstream filter missing: ") + bsf_name);
    }
    if (av_bsf_alloc(filter, &ff_->bsf) < 0) {
        close_input();
        return Status::Fail("av_bsf_alloc failed");
    }
    if (avcodec_parameters_copy(ff_->bsf->par_in, par) < 0) {
        close_input();
        return Status::Fail("avcodec_parameters_copy failed");
    }
    ff_->bsf->time_base_in = stream->time_base;
    const int bsf_ret = av_bsf_init(ff_->bsf);
    if (bsf_ret < 0) {
        close_input();
        return Status::Fail("av_bsf_init: " + ffmpeg_err(bsf_ret));
    }

    ff_->video_index = video_index;
    ff_->time_base = stream->time_base;
    width_ = par->width;
    height_ = par->height;
    YR_LOGI("video %s %dx%d stream=%d (demux only, no software decoder)", codec_name_.c_str(), width_,
            height_, video_index);
    return Status::Ok();
}

ReadStatus RtspClient::read(EncodedPacket& out) {
    if (!pending_.empty()) {
        out = std::move(pending_.front());
        pending_.pop_front();
        return ReadStatus::Packet;
    }
    if (ff_ == nullptr || ff_->fmt == nullptr || ff_->bsf == nullptr) {
        return ReadStatus::Error;
    }

    while (!(stop_ && stop_->load())) {
        AVPacket packet;
        av_init_packet(&packet);
        packet.data = nullptr;
        packet.size = 0;
        const int got = av_read_frame(ff_->fmt, &packet);
        if (got < 0) {
            av_packet_unref(&packet);
            if (stop_ && stop_->load()) {
                return ReadStatus::Stopped;
            }
            if (got == AVERROR_EOF) {
                return ReadStatus::Eos;
            }
            YR_LOGW("av_read_frame: %s", ffmpeg_err(got).c_str());
            return ReadStatus::Error;
        }
        if (packet.stream_index != ff_->video_index) {
            av_packet_unref(&packet);
            continue;
        }

        // 成功后 packet 的所有权交给 bitstream filter，不能再 unref。
        const int sent = av_bsf_send_packet(ff_->bsf, &packet);
        if (sent < 0) {
            av_packet_unref(&packet);
            YR_LOGW("av_bsf_send_packet: %s", ffmpeg_err(sent).c_str());
            return ReadStatus::Error;
        }

        while (true) {
            AVPacket annexb;
            av_init_packet(&annexb);
            annexb.data = nullptr;
            annexb.size = 0;
            const int rec = av_bsf_receive_packet(ff_->bsf, &annexb);
            if (rec == AVERROR(EAGAIN) || rec == AVERROR_EOF) {
                av_packet_unref(&annexb);
                break;
            }
            if (rec < 0) {
                av_packet_unref(&annexb);
                YR_LOGW("av_bsf_receive_packet: %s", ffmpeg_err(rec).c_str());
                return ReadStatus::Error;
            }
            EncodedPacket encoded;
            if (annexb.data != nullptr && annexb.size > 0) {
                encoded.data.assign(annexb.data, annexb.data + annexb.size);
            }
            if (annexb.pts != AV_NOPTS_VALUE) {
                encoded.pts_us = av_rescale_q(annexb.pts, ff_->time_base, av_make_q(1, 1000000));
            }
            encoded.keyframe = (annexb.flags & AV_PKT_FLAG_KEY) != 0;
            av_packet_unref(&annexb);
            pending_.push_back(std::move(encoded));
        }
        if (!pending_.empty()) {
            out = std::move(pending_.front());
            pending_.pop_front();
            return ReadStatus::Packet;
        }
    }
    return ReadStatus::Stopped;
}

}  // namespace yolo_rknn
