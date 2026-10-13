#define MODULE_TAG "yr_mpp"

#include "yolo_rknn/decode/mpp_decoder.hpp"

#include "yolo_rknn/common/logger.hpp"
#include "yolo_rknn/decode/mpp_c_api.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

namespace yolo_rknn {

struct DecodedFrame::Retain {
    MppFrame frame = nullptr;
    explicit Retain(MppFrame f) : frame(f) {}
    ~Retain() {
        if (frame) {
            mpp_frame_deinit(&frame);
        }
    }
};

struct MppDecoder::Impl {
    MppCtx ctx = nullptr;
    MppApi* mpi = nullptr;
    MppBufferGroup group = nullptr;
    // 分辨率切换后，旧 group 里的 buffer 可能还被队列中的帧引用，不能立刻 put。
    std::vector<MppBufferGroup> retired;
    bool inited = false;
};

const char* video_codec_name(VideoCodec codec) {
    return codec == VideoCodec::H265 ? "h265" : "h264";
}

MppDecoder::MppDecoder(const AppConfig& config) : config_(config), impl_(new Impl) {}

MppDecoder::~MppDecoder() {
    if (impl_ == nullptr) {
        return;
    }
    // 先拆解码器，让它放下自己还握着的 buffer，再释放 group。
    // 调用方必须已经销毁所有 DecodedFrame，否则这里会碰到仍在使用的 DRM buffer。
    if (impl_->ctx) {
        mpp_destroy(impl_->ctx);
        impl_->ctx = nullptr;
        impl_->mpi = nullptr;
    }
    if (impl_->group) {
        mpp_buffer_group_put(impl_->group);
        impl_->group = nullptr;
    }
    for (std::size_t i = 0; i < impl_->retired.size(); ++i) {
        if (impl_->retired[i]) {
            mpp_buffer_group_put(impl_->retired[i]);
        }
    }
    delete impl_;
    impl_ = nullptr;
}

Status MppDecoder::init(VideoCodec codec) {
    if (impl_->inited) {
        return Status::Ok();
    }
    const MppCodingType coding =
        (codec == VideoCodec::H265) ? MPP_VIDEO_CodingHEVC : MPP_VIDEO_CodingAVC;
    if (mpp_check_support_format(MPP_CTX_DEC, coding) != MPP_OK) {
        return Status::Fail(std::string("VPU does not list support for ") + video_codec_name(codec));
    }

    MPP_RET ret = mpp_create(&impl_->ctx, &impl_->mpi);
    if (ret != MPP_OK || impl_->ctx == nullptr || impl_->mpi == nullptr) {
        return Status::Fail("mpp_create failed");
    }

    // split / fast 必须在 mpp_init 之前。split 让 VPU 自己切 NAL，
    // 这样一包里同时带 SPS、PPS、IDR 也不会被整包丢掉。
    RK_U32 split = 1;
    ret = impl_->mpi->control(impl_->ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &split);
    if (ret != MPP_OK) {
        YR_LOGW("MPP_DEC_SET_PARSER_SPLIT_MODE -> %d", ret);
    }
    if (config_.low_latency_decode) {
        RK_U32 fast = 1;
        ret = impl_->mpi->control(impl_->ctx, MPP_DEC_SET_PARSER_FAST_MODE, &fast);
        if (ret != MPP_OK) {
            YR_LOGW("MPP_DEC_SET_PARSER_FAST_MODE -> %d", ret);
        }
    }

    ret = mpp_init(impl_->ctx, MPP_CTX_DEC, coding);
    if (ret != MPP_OK) {
        return Status::Fail("mpp_init failed: " + std::to_string(ret));
    }

    MppFrameFormat nv12 = MPP_FMT_YUV420SP;
    ret = impl_->mpi->control(impl_->ctx, MPP_DEC_SET_OUTPUT_FORMAT, &nv12);
    if (ret != MPP_OK) {
        YR_LOGW("MPP_DEC_SET_OUTPUT_FORMAT NV12 -> %d", ret);
    }
    if (config_.low_latency_decode) {
        RK_U32 immediate = 1;
        ret = impl_->mpi->control(impl_->ctx, MPP_DEC_SET_IMMEDIATE_OUT, &immediate);
        if (ret != MPP_OK) {
            YR_LOGW("MPP_DEC_SET_IMMEDIATE_OUT -> %d", ret);
        }
    }

    impl_->inited = true;
    YR_LOGI("VPU decoder ready codec=%s low_latency=%d", video_codec_name(codec),
            config_.low_latency_decode ? 1 : 0);
    return Status::Ok();
}

Status MppDecoder::ensure_buffer_group(void* frame_ptr) {
    MppFrame frame = frame_ptr;
    const RK_U32 width = mpp_frame_get_width(frame);
    const RK_U32 height = mpp_frame_get_height(frame);
    const RK_U32 hor = mpp_frame_get_hor_stride(frame);
    const RK_U32 ver = mpp_frame_get_ver_stride(frame);
    size_t buf_size = mpp_frame_get_buf_size(frame);
    if (buf_size == 0) {
        buf_size = static_cast<size_t>(hor) * static_cast<size_t>(ver) * 3 / 2;
    }

    if (impl_->group) {
        impl_->retired.push_back(impl_->group);
        impl_->group = nullptr;
    }

    // DRM + CONTIG：内核从 CMA 拿物理连续内存，相当于这块板子上的 “MMZ”。
    // 不申请 NORMAL（malloc）缓冲，那种内存 RGA 不能用 fd 导入，只能 CPU 搬。
    const MppBufferType type =
        static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG);
    MppBufferGroup group = nullptr;
    MPP_RET ret = mpp_buffer_group_get_internal(&group, type);
    if (ret != MPP_OK || group == nullptr) {
        return Status::Fail("DRM contiguous buffer group alloc failed: " + std::to_string(ret));
    }
    ret = mpp_buffer_group_limit_config(group, buf_size, config_.decoder_buf_count);
    if (ret != MPP_OK) {
        mpp_buffer_group_put(group);
        return Status::Fail("mpp_buffer_group_limit_config failed");
    }
    ret = impl_->mpi->control(impl_->ctx, MPP_DEC_SET_EXT_BUF_GROUP, group);
    if (ret != MPP_OK) {
        mpp_buffer_group_put(group);
        return Status::Fail("MPP_DEC_SET_EXT_BUF_GROUP failed");
    }
    ret = impl_->mpi->control(impl_->ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
    if (ret != MPP_OK) {
        mpp_buffer_group_put(group);
        return Status::Fail("MPP_DEC_SET_INFO_CHANGE_READY failed");
    }
    impl_->group = group;
    YR_LOGI("NV12 DRM buffers %ux%u stride %ux%u size=%zu count=%d", width, height, hor, ver, buf_size,
            config_.decoder_buf_count);
    return Status::Ok();
}

Status MppDecoder::drain(std::vector<DecodedFrame>& frames) {
    while (true) {
        MppFrame frame = nullptr;
        const MPP_RET ret = impl_->mpi->decode_get_frame(impl_->ctx, &frame);
        if (ret != MPP_OK || frame == nullptr) {
            break;
        }
        if (mpp_frame_get_info_change(frame)) {
            Status st = ensure_buffer_group(frame);
            mpp_frame_deinit(&frame);
            if (!st.ok()) {
                return st;
            }
            continue;
        }
        if (mpp_frame_get_eos(frame)) {
            eos_seen_ = true;
            mpp_frame_deinit(&frame);
            break;
        }
        if (mpp_frame_get_errinfo(frame) || mpp_frame_get_discard(frame)) {
            mpp_frame_deinit(&frame);
            continue;
        }

        const RK_U32 fmt = static_cast<RK_U32>(mpp_frame_get_fmt(frame));
        const RK_U32 family = fmt & MPP_FRAME_FMT_MASK;
        const RK_U32 id = fmt & ~MPP_FRAME_FMT_MASK;
        if (family != MPP_FRAME_FMT_YUV || id != static_cast<RK_U32>(MPP_FMT_YUV420SP)) {
            if (!warned_format_) {
                YR_LOGE("VPU frame format %#x is not NV12. Drop it. CPU color conversion is not available",
                        fmt);
                warned_format_ = true;
            }
            mpp_frame_deinit(&frame);
            continue;
        }

        MppBuffer buffer = mpp_frame_get_buffer(frame);
        if (buffer == nullptr) {
            mpp_frame_deinit(&frame);
            continue;
        }
        const int fd = mpp_buffer_get_fd(buffer);
        if (fd < 0) {
            YR_LOGE("NV12 frame has no dma-buf fd");
            mpp_frame_deinit(&frame);
            continue;
        }

        DecodedFrame decoded;
        decoded.fd = fd;
        decoded.width = static_cast<int>(mpp_frame_get_width(frame));
        decoded.height = static_cast<int>(mpp_frame_get_height(frame));
        decoded.hor_stride = static_cast<int>(mpp_frame_get_hor_stride(frame));
        decoded.ver_stride = static_cast<int>(mpp_frame_get_ver_stride(frame));
        decoded.pts_us = mpp_frame_get_pts(frame);
        decoded.seq = seq_++;
        decoded.retain = std::make_shared<DecodedFrame::Retain>(frame);
        frames.push_back(std::move(decoded));
    }
    return Status::Ok();
}

Status MppDecoder::decode(const EncodedPacket& packet, std::vector<DecodedFrame>& frames) {
    if (!impl_->inited || impl_->mpi == nullptr) {
        return Status::Fail("decoder is not initialized");
    }
    if (packet.data.empty()) {
        return Status::Ok();
    }

    // mpp_packet_init 不拷贝码流。decode_put_packet 返回成功后解码器已把数据收走，
    // 这里的 vector 只需要在这次调用期间活着。拷贝的是压缩包，不是 NV12。
    MppPacket mpkt = nullptr;
    MPP_RET ret = mpp_packet_init(&mpkt, const_cast<std::uint8_t*>(packet.data.data()), packet.data.size());
    if (ret != MPP_OK || mpkt == nullptr) {
        return Status::Fail("mpp_packet_init failed");
    }
    if (packet.pts_us >= 0) {
        mpp_packet_set_pts(mpkt, packet.pts_us);
    }

    bool sent = false;
    for (int attempt = 0; attempt < 30; ++attempt) {
        ret = impl_->mpi->decode_put_packet(impl_->ctx, mpkt);
        if (ret == MPP_OK) {
            sent = true;
            break;
        }
        if (ret == MPP_ERR_BUFFER_FULL) {
            Status drained = drain(frames);
            if (!drained.ok()) {
                mpp_packet_deinit(&mpkt);
                return drained;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        mpp_packet_deinit(&mpkt);
        return Status::Fail("decode_put_packet failed: " + std::to_string(ret));
    }
    mpp_packet_deinit(&mpkt);
    if (!sent) {
        return Status::Fail("VPU input stayed full");
    }
    return drain(frames);
}

Status MppDecoder::flush(std::vector<DecodedFrame>& frames) {
    if (!impl_->inited || impl_->mpi == nullptr) {
        return Status::Ok();
    }
    eos_seen_ = false;
    MppPacket mpkt = nullptr;
    MPP_RET ret = mpp_packet_init(&mpkt, nullptr, 0);
    if (ret != MPP_OK || mpkt == nullptr) {
        return Status::Fail("mpp_packet_init eos failed");
    }
    mpp_packet_set_eos(mpkt);
    bool sent = false;
    for (int attempt = 0; attempt < 10; ++attempt) {
        ret = impl_->mpi->decode_put_packet(impl_->ctx, mpkt);
        if (ret == MPP_OK) {
            sent = true;
            break;
        }
        if (ret == MPP_ERR_BUFFER_FULL) {
            Status drained = drain(frames);
            if (!drained.ok()) {
                mpp_packet_deinit(&mpkt);
                return drained;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        mpp_packet_deinit(&mpkt);
        return Status::Fail("eos packet rejected: " + std::to_string(ret));
    }
    mpp_packet_deinit(&mpkt);
    if (!sent) {
        return Status::Fail("VPU did not accept eos");
    }
    for (int attempt = 0; attempt < 30 && !eos_seen_; ++attempt) {
        const std::size_t before = frames.size();
        Status drained = drain(frames);
        if (!drained.ok()) {
            return drained;
        }
        if (eos_seen_ || frames.size() == before) {
            if (eos_seen_) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    return Status::Ok();
}

}  // namespace yolo_rknn
