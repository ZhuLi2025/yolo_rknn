#define MODULE_TAG "yr_rec"

#include "yolo_rknn/record/mp4_recorder.hpp"

#include "yolo_rknn/common/dma_buf.hpp"
#include "yolo_rknn/common/logger.hpp"
#include "yolo_rknn/decode/mpp_c_api.hpp"
#include "yolo_rknn/display/osd_painter.hpp"
#include "yolo_rknn/preprocess/rga_lock.hpp"

#include "im2d.h"
#include "rk_venc_cfg.h"

extern "C" {
#include "rk_venc_cmd.h"
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace yolo_rknn {
namespace {

constexpr int kFps = 25;
constexpr int kBitrate = 8000000;
constexpr int kGop = kFps * 2;

struct DmaSlot {
    MppBufferGroup group = nullptr;
    MppBuffer buffer = nullptr;
    int fd = -1;
    void* ptr = nullptr;
    std::size_t bytes = 0;
    bool cacheable = false;
};

struct Nal {
    const std::uint8_t* data = nullptr;
    int size = 0;
    int type = 0;
};

int align16(int value) { return (value + 15) & ~15; }

bool rga_ok(IM_STATUS status) { return status == IM_STATUS_SUCCESS || status == IM_STATUS_NOERROR; }

const char* rga_error(IM_STATUS status) {
    const char* text = imStrError_t(status);
    return text != nullptr ? text : "unknown";
}

std::string ffmpeg_err(int err) {
    char buf[160];
    av_strerror(err, buf, sizeof(buf));
    return std::string(buf);
}

void release_slot(DmaSlot* slot) {
    if (slot == nullptr) {
        return;
    }
    if (slot->buffer) {
        mpp_buffer_put(slot->buffer);
        slot->buffer = nullptr;
    }
    if (slot->group) {
        mpp_buffer_group_put(slot->group);
        slot->group = nullptr;
    }
    slot->fd = -1;
    slot->ptr = nullptr;
    slot->bytes = 0;
}

Status alloc_slot(DmaSlot* slot, std::size_t bytes, bool prefer_cacheable, const char** allocator_name) {
    struct Candidate {
        MppBufferType type;
        const char* name;
        bool cacheable;
    };
    const Candidate all[] = {
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG), "DRM CMA", false},
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CONTIG), "dma-heap CMA", false},
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG | MPP_BUFFER_FLAGS_CACHABLE),
         "DRM CMA cacheable", true},
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CONTIG | MPP_BUFFER_FLAGS_CACHABLE),
         "dma-heap CMA cacheable", true},
    };
    const int order_cached[] = {2, 3, 0, 1};
    const int order_plain[] = {0, 1, 2, 3};
    const int* order = prefer_cacheable ? order_cached : order_plain;
    for (int n = 0; n < 4; ++n) {
        const Candidate& candidate = all[order[n]];
        MppBufferGroup trial = nullptr;
        MPP_RET ret = mpp_buffer_group_get_internal(&trial, candidate.type);
        if (ret != MPP_OK || trial == nullptr) {
            continue;
        }
        MppBuffer trial_buf = nullptr;
        ret = mpp_buffer_get(trial, &trial_buf, bytes);
        if (ret != MPP_OK || trial_buf == nullptr) {
            mpp_buffer_group_put(trial);
            continue;
        }
        void* ptr = mpp_buffer_get_ptr(trial_buf);
        const int fd = mpp_buffer_get_fd(trial_buf);
        if (ptr == nullptr || fd < 0 || mpp_buffer_get_size(trial_buf) < bytes) {
            mpp_buffer_put(trial_buf);
            mpp_buffer_group_put(trial);
            continue;
        }
        slot->group = trial;
        slot->buffer = trial_buf;
        slot->ptr = ptr;
        slot->fd = fd;
        slot->bytes = bytes;
        slot->cacheable = candidate.cacheable;
        if (allocator_name != nullptr) {
            *allocator_name = candidate.name;
        }
        return Status::Ok();
    }
    return Status::Fail("record buffer alloc failed");
}

void cpu_write_begin(const DmaSlot& slot) {
    if (slot.cacheable && slot.fd >= 0) {
        dma_buf_begin_cpu_write(slot.fd);
    }
}

void cpu_write_end(const DmaSlot& slot) {
    if (slot.cacheable && slot.fd >= 0) {
        dma_buf_end_cpu_write(slot.fd);
    }
}

void collect_nals(const std::uint8_t* data, int size, std::vector<Nal>* out) {
    out->clear();
    if (data == nullptr || size < 4) {
        return;
    }
    int i = 0;
    while (i + 3 < size) {
        int sc = 0;
        if (i + 4 <= size && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1) {
            sc = 4;
        } else if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
            sc = 3;
        } else {
            ++i;
            continue;
        }
        const int start = i + sc;
        int next = start;
        while (next + 3 < size) {
            const bool four = next + 4 <= size && data[next] == 0 && data[next + 1] == 0 && data[next + 2] == 0 &&
                              data[next + 3] == 1;
            const bool three = data[next] == 0 && data[next + 1] == 0 && data[next + 2] == 1;
            if (four || three) {
                break;
            }
            ++next;
        }
        if (next + 3 >= size) {
            next = size;
        }
        int nal_size = next - start;
        while (nal_size > 0 && data[start + nal_size - 1] == 0) {
            --nal_size;
        }
        if (nal_size > 0) {
            Nal nal;
            nal.data = data + start;
            nal.size = nal_size;
            nal.type = data[start] & 0x1f;
            out->push_back(nal);
        }
        i = next;
    }
}

void append_be16(std::vector<std::uint8_t>* out, int value) {
    out->push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
    out->push_back(static_cast<std::uint8_t>(value & 0xff));
}

void append_be32(std::vector<std::uint8_t>* out, int value) {
    out->push_back(static_cast<std::uint8_t>((value >> 24) & 0xff));
    out->push_back(static_cast<std::uint8_t>((value >> 16) & 0xff));
    out->push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
    out->push_back(static_cast<std::uint8_t>(value & 0xff));
}

Status avcc_extradata(const std::vector<Nal>& nals, std::vector<std::uint8_t>* out) {
    const Nal* sps = nullptr;
    const Nal* pps = nullptr;
    for (std::size_t i = 0; i < nals.size(); ++i) {
        if (nals[i].type == 7 && sps == nullptr) {
            sps = &nals[i];
        } else if (nals[i].type == 8 && pps == nullptr) {
            pps = &nals[i];
        }
    }
    if (sps == nullptr || pps == nullptr || sps->size < 4 || pps->size < 1) {
        return Status::Fail("encoder header has no SPS/PPS");
    }
    out->clear();
    out->push_back(1);
    out->push_back(sps->data[1]);
    out->push_back(sps->data[2]);
    out->push_back(sps->data[3]);
    out->push_back(0xff);
    out->push_back(0xe1);
    append_be16(out, sps->size);
    out->insert(out->end(), sps->data, sps->data + sps->size);
    out->push_back(1);
    append_be16(out, pps->size);
    out->insert(out->end(), pps->data, pps->data + pps->size);
    return Status::Ok();
}

bool packet_avcc(const std::uint8_t* data, int size, std::vector<std::uint8_t>* out, bool* keyframe) {
    std::vector<Nal> nals;
    collect_nals(data, size, &nals);
    out->clear();
    *keyframe = false;
    for (std::size_t i = 0; i < nals.size(); ++i) {
        const int type = nals[i].type;
        if (type == 6 || type == 7 || type == 8 || type == 9) {
            continue;
        }
        if (type == 5) {
            *keyframe = true;
        }
        append_be32(out, nals[i].size);
        out->insert(out->end(), nals[i].data, nals[i].data + nals[i].size);
    }
    return !out->empty();
}

IM_STATUS convert_fd(int src_fd, int src_w, int src_h, int src_ws, int src_hs, int src_fmt, int dst_fd, int dst_w,
                     int dst_h, int dst_ws, int dst_hs, int dst_fmt) {
    rga_buffer_t src = wrapbuffer_fd_t(src_fd, src_w, src_h, src_ws, src_hs, src_fmt);
    rga_buffer_t dst = wrapbuffer_fd_t(dst_fd, dst_w, dst_h, dst_ws, dst_hs, dst_fmt);
    rga_buffer_t pat;
    std::memset(&pat, 0, sizeof(pat));
    im_rect src_rect;
    src_rect.x = 0;
    src_rect.y = 0;
    src_rect.width = src_w;
    src_rect.height = src_h;
    im_rect dst_rect;
    dst_rect.x = 0;
    dst_rect.y = 0;
    dst_rect.width = dst_w;
    dst_rect.height = dst_h;
    im_rect pat_rect;
    std::memset(&pat_rect, 0, sizeof(pat_rect));
    return improcess(src, dst, pat, src_rect, dst_rect, pat_rect, -1, nullptr, nullptr, IM_SYNC);
}

Status set_cfg_key(MppEncCfg cfg, const char* key, int value) {
    const MPP_RET ret = mpp_enc_cfg_set_s32(cfg, key, value);
    if (ret != MPP_OK) {
        return Status::Fail(std::string("mpp enc cfg ") + key + " -> " + std::to_string(ret));
    }
    return Status::Ok();
}

std::string make_path(const std::string& dir) {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf;
    localtime_r(&now, &tm_buf);
    char stamp[32];
    if (std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &tm_buf) == 0) {
        std::snprintf(stamp, sizeof(stamp), "record");
    }
    const std::string slash = (!dir.empty() && dir.back() == '/') ? "" : "/";
    std::string path = dir + slash + stamp + ".mp4";
    if (access(path.c_str(), F_OK) != 0) {
        return path;
    }
    for (int n = 2; n < 100; ++n) {
        path = dir + slash + stamp + "_" + std::to_string(n) + ".mp4";
        if (access(path.c_str(), F_OK) != 0) {
            return path;
        }
    }
    return dir + slash + stamp + "_x.mp4";
}

}  // namespace

struct Mp4Recorder::Impl {
    std::string path;
    bool prepared = false;
    bool opened = false;
    bool file_created = false;
    int width = 0;
    int height = 0;
    int hor = 0;
    int ver = 0;
    DmaSlot video;
    DmaSlot osd;
    DmaSlot nv12;
    MppCtx ctx = nullptr;
    MppApi* mpi = nullptr;
    MppEncCfg cfg = nullptr;
    MppFrame enc_frame = nullptr;
    AVFormatContext* oc = nullptr;
    AVStream* stream = nullptr;
    std::int64_t origin_pts = -1;
    std::int64_t last_pts = 0;
    std::uint64_t frames = 0;
    bool blend_failed = false;
    bool rga_warned = false;
    bool empty_warned = false;
};

Mp4Recorder::Mp4Recorder(const AppConfig& config) : config_(config), impl_(new Impl) {}

Mp4Recorder::~Mp4Recorder() {
    close();
    delete impl_;
    impl_ = nullptr;
}

const std::string& Mp4Recorder::path() const { return impl_->path; }

std::uint64_t Mp4Recorder::frames() const { return impl_->frames; }

Status Mp4Recorder::prepare() {
    if (impl_->prepared) {
        return Status::Ok();
    }
    struct stat st;
    if (stat(config_.record_dir.c_str(), &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
            return Status::Fail("record_dir is not a directory: " + config_.record_dir);
        }
    } else if (mkdir(config_.record_dir.c_str(), 0755) != 0) {
        return Status::Fail(std::string("cannot create ") + config_.record_dir + ": " + std::strerror(errno));
    }
    impl_->path = make_path(config_.record_dir);
    impl_->prepared = true;
    YR_LOGI("record file %s", impl_->path.c_str());
    return Status::Ok();
}

void Mp4Recorder::close() {
    if (impl_ == nullptr) {
        return;
    }
    if (impl_->opened && impl_->mpi != nullptr && impl_->ctx != nullptr && impl_->enc_frame != nullptr &&
        impl_->nv12.buffer != nullptr) {
        RK_S64 drain_ms = 200;
        impl_->mpi->control(impl_->ctx, MPP_SET_OUTPUT_TIMEOUT, &drain_ms);
        mpp_frame_set_eos(impl_->enc_frame, 1);
        MppPacket packet = nullptr;
        const MPP_RET enc = impl_->mpi->encode_put_frame(impl_->ctx, impl_->enc_frame);
        if (enc == MPP_OK) {
            const MPP_RET got = impl_->mpi->encode_get_packet(impl_->ctx, &packet);
            if (got == MPP_OK) {
                const Status flushed = write_packet(packet, impl_->last_pts);
                if (!flushed.ok()) {
                    YR_LOGW("%s", flushed.message().c_str());
                }
            }
        }
    }
    if (impl_->enc_frame) {
        mpp_frame_deinit(&impl_->enc_frame);
        impl_->enc_frame = nullptr;
    }
    if (impl_->oc != nullptr) {
        if (impl_->frames > 0) {
            av_write_trailer(impl_->oc);
        }
        if (impl_->oc->pb != nullptr) {
            avio_closep(&impl_->oc->pb);
        }
        avformat_free_context(impl_->oc);
        impl_->oc = nullptr;
        impl_->stream = nullptr;
    }
    if (impl_->file_created && impl_->frames == 0 && !impl_->path.empty()) {
        unlink(impl_->path.c_str());
        impl_->file_created = false;
    }
    if (impl_->ctx) {
        mpp_destroy(impl_->ctx);
        impl_->ctx = nullptr;
        impl_->mpi = nullptr;
    }
    if (impl_->cfg) {
        mpp_enc_cfg_deinit(impl_->cfg);
        impl_->cfg = nullptr;
    }
    release_slot(&impl_->video);
    release_slot(&impl_->osd);
    release_slot(&impl_->nv12);
    if (impl_->opened) {
        YR_LOGI("record closed frames=%llu path=%s", static_cast<unsigned long long>(impl_->frames),
                impl_->path.c_str());
    }
    impl_->opened = false;
}

Status Mp4Recorder::write(const RecordSample& sample) {
    if (!impl_->prepared) {
        return Status::Fail("recorder is not prepared");
    }
    const DecodedFrame& frame = sample.frame;
    const int width = frame.width & ~1;
    const int height = frame.height & ~1;
    if (frame.fd < 0 || width < 16 || height < 16) {
        return Status::Ok();
    }
    if (!impl_->opened) {
        const int hor = align16(width);
        const int ver = align16(height);
        const std::size_t bgra_bytes = static_cast<std::size_t>(hor) * static_cast<std::size_t>(ver) * 4;
        const std::size_t nv12_bytes = static_cast<std::size_t>(hor) * static_cast<std::size_t>(ver) * 3 / 2;
        const char* video_name = "";
        const char* osd_name = "";
        const char* nv12_name = "";
        Status st = alloc_slot(&impl_->video, bgra_bytes, false, &video_name);
        if (!st.ok()) {
            return st;
        }
        st = alloc_slot(&impl_->osd, bgra_bytes, true, &osd_name);
        if (!st.ok()) {
            release_slot(&impl_->video);
            return st;
        }
        st = alloc_slot(&impl_->nv12, nv12_bytes, false, &nv12_name);
        if (!st.ok()) {
            release_slot(&impl_->video);
            release_slot(&impl_->osd);
            return st;
        }
        cpu_write_begin(impl_->nv12);
        std::memset(impl_->nv12.ptr, 0, nv12_bytes);
        cpu_write_end(impl_->nv12);

        MPP_RET ret = mpp_create(&impl_->ctx, &impl_->mpi);
        if (ret != MPP_OK || impl_->ctx == nullptr || impl_->mpi == nullptr) {
            close();
            return Status::Fail("mpp encoder create failed");
        }
        ret = mpp_init(impl_->ctx, MPP_CTX_ENC, MPP_VIDEO_CodingAVC);
        if (ret != MPP_OK) {
            close();
            return Status::Fail("mpp encoder init failed: " + std::to_string(ret));
        }
        // 不设阻塞超时的话，encode() 会立刻返回空包，码流还在编码器里。
        RK_S64 timeout_ms = 2000;
        impl_->mpi->control(impl_->ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout_ms);
        impl_->mpi->control(impl_->ctx, MPP_SET_INPUT_TIMEOUT, &timeout_ms);
        ret = mpp_enc_cfg_init(&impl_->cfg);
        if (ret != MPP_OK || impl_->cfg == nullptr) {
            close();
            return Status::Fail("mpp_enc_cfg_init failed");
        }
        const int profile_try[] = {100, 66};
        const int cabac_try[] = {1, 0};
        const int trans_try[] = {1, 0};
        bool configured = false;
        for (int attempt = 0; attempt < 2 && !configured; ++attempt) {
            st = set_cfg_key(impl_->cfg, "prep:width", width);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "prep:height", height);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "prep:hor_stride", hor);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "prep:ver_stride", ver);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "prep:format", MPP_FMT_YUV420SP);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "rc:mode", MPP_ENC_RC_MODE_CBR);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "rc:bps_target", kBitrate);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "rc:bps_max", kBitrate * 17 / 16);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "rc:bps_min", kBitrate * 15 / 16);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "rc:fps_in_num", kFps);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "rc:fps_in_denorm", 1);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "rc:fps_out_num", kFps);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "rc:fps_out_denorm", 1);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "rc:gop", kGop);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "codec:type", MPP_VIDEO_CodingAVC);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "h264:profile", profile_try[attempt]);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "h264:level", 40);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "h264:cabac_en", cabac_try[attempt]);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "h264:trans8x8", trans_try[attempt]);
            if (!st.ok()) {
                close();
                return st;
            }
            st = set_cfg_key(impl_->cfg, "h264:stream_type", 0);
            if (!st.ok()) {
                close();
                return st;
            }
            ret = impl_->mpi->control(impl_->ctx, MPP_ENC_SET_CFG, impl_->cfg);
            if (ret == MPP_OK) {
                configured = true;
                if (attempt == 1) {
                    YR_LOGW("h264 high profile rejected, using baseline");
                }
            }
        }
        if (!configured) {
            close();
            return Status::Fail("MPP_ENC_SET_CFG failed: " + std::to_string(ret));
        }
        MppEncHeaderMode header_mode = MPP_ENC_HEADER_MODE_DEFAULT;
        impl_->mpi->control(impl_->ctx, MPP_ENC_SET_HEADER_MODE, &header_mode);

        std::uint8_t header_buf[4096];
        std::memset(header_buf, 0, sizeof(header_buf));
        MppPacket header = nullptr;
        ret = mpp_packet_init(&header, header_buf, sizeof(header_buf));
        if (ret != MPP_OK) {
            close();
            return Status::Fail("mpp header packet init failed");
        }
        mpp_packet_set_length(header, 0);
        ret = impl_->mpi->control(impl_->ctx, MPP_ENC_GET_HDR_SYNC, header);
        const std::uint8_t* header_pos = static_cast<const std::uint8_t*>(mpp_packet_get_pos(header));
        const int header_len = static_cast<int>(mpp_packet_get_length(header));
        std::vector<Nal> header_nals;
        collect_nals(header_pos, header_len, &header_nals);
        std::vector<std::uint8_t> extra;
        st = avcc_extradata(header_nals, &extra);
        mpp_packet_deinit(&header);
        if (!st.ok()) {
            close();
            return st;
        }

        int averr = avformat_alloc_output_context2(&impl_->oc, nullptr, "mp4", impl_->path.c_str());
        if (averr < 0 || impl_->oc == nullptr) {
            close();
            return Status::Fail("mp4 muxer alloc failed: " + ffmpeg_err(averr));
        }
        impl_->stream = avformat_new_stream(impl_->oc, nullptr);
        if (impl_->stream == nullptr) {
            close();
            return Status::Fail("mp4 stream alloc failed");
        }
        impl_->stream->time_base = av_make_q(1, 1000000);
        impl_->stream->avg_frame_rate = av_make_q(kFps, 1);
        impl_->stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
        impl_->stream->codecpar->codec_id = AV_CODEC_ID_H264;
        impl_->stream->codecpar->width = width;
        impl_->stream->codecpar->height = height;
        impl_->stream->codecpar->format = AV_PIX_FMT_YUV420P;
        impl_->stream->codecpar->bit_rate = kBitrate;
        impl_->stream->codecpar->extradata_size = static_cast<int>(extra.size());
        impl_->stream->codecpar->extradata = static_cast<std::uint8_t*>(
            av_mallocz(extra.size() + static_cast<std::size_t>(AV_INPUT_BUFFER_PADDING_SIZE)));
        if (impl_->stream->codecpar->extradata == nullptr) {
            close();
            return Status::Fail("mp4 extradata alloc failed");
        }
        std::memcpy(impl_->stream->codecpar->extradata, extra.data(), extra.size());
        averr = avio_open(&impl_->oc->pb, impl_->path.c_str(), AVIO_FLAG_WRITE);
        if (averr < 0) {
            close();
            return Status::Fail("cannot open " + impl_->path + ": " + ffmpeg_err(averr));
        }
        impl_->file_created = true;
        averr = avformat_write_header(impl_->oc, nullptr);
        if (averr < 0) {
            close();
            return Status::Fail("mp4 header failed: " + ffmpeg_err(averr));
        }

        ret = mpp_frame_init(&impl_->enc_frame);
        if (ret != MPP_OK || impl_->enc_frame == nullptr) {
            close();
            return Status::Fail("mpp_frame_init failed");
        }
        mpp_frame_set_width(impl_->enc_frame, width);
        mpp_frame_set_height(impl_->enc_frame, height);
        mpp_frame_set_hor_stride(impl_->enc_frame, hor);
        mpp_frame_set_ver_stride(impl_->enc_frame, ver);
        mpp_frame_set_fmt(impl_->enc_frame, MPP_FMT_YUV420SP);
        mpp_frame_set_buffer(impl_->enc_frame, impl_->nv12.buffer);
        mpp_frame_set_buf_size(impl_->enc_frame, nv12_bytes);

        impl_->width = width;
        impl_->height = height;
        impl_->hor = hor;
        impl_->ver = ver;
        impl_->opened = true;
        YR_LOGI("record %dx%d stride %dx%d h264 %d kbps video=%s osd=%s nv12=%s", width, height, hor, ver,
                kBitrate / 1000, video_name, osd_name, nv12_name);
    }
    if (width != impl_->width || height != impl_->height) {
        YR_LOGW("record keeps %dx%d, skip %dx%d", impl_->width, impl_->height, width, height);
        return Status::Ok();
    }

    const int src_ws = frame.hor_stride > 0 ? frame.hor_stride : width;
    const int src_hs = frame.ver_stride > 0 ? frame.ver_stride : height;
    IM_STATUS converted = IM_STATUS_FAILED;
    {
        RgaApiLock lock;
        converted = convert_fd(frame.fd, width, height, src_ws, src_hs, RK_FORMAT_YCbCr_420_SP, impl_->video.fd, width,
                               height, impl_->hor, impl_->ver, RK_FORMAT_BGRA_8888);
    }
    if (!rga_ok(converted)) {
        if (!impl_->rga_warned) {
            YR_LOGW("record RGA NV12->BGRA failed: %s", rga_error(converted));
            impl_->rga_warned = true;
        }
        return Status::Ok();
    }

    cpu_write_begin(impl_->osd);
    std::memset(impl_->osd.ptr, 0, impl_->osd.bytes);
    OsdCanvas canvas;
    canvas.pixels = static_cast<std::uint8_t*>(impl_->osd.ptr);
    canvas.width = width;
    canvas.height = height;
    canvas.stride_bytes = impl_->hor * 4;
    osd_draw(canvas, sample.result.detections);
    cpu_write_end(impl_->osd);

    if (!impl_->blend_failed) {
        rga_buffer_t fg = wrapbuffer_fd_t(impl_->osd.fd, width, height, impl_->hor, impl_->ver, RK_FORMAT_BGRA_8888);
        rga_buffer_t bg = wrapbuffer_fd_t(impl_->video.fd, width, height, impl_->hor, impl_->ver, RK_FORMAT_BGRA_8888);
        IM_STATUS blended = IM_STATUS_FAILED;
        {
            RgaApiLock lock;
            blended = imblend_t(fg, bg, bg, IM_ALPHA_BLEND_SRC_OVER, 1);
        }
        if (!rga_ok(blended)) {
            impl_->blend_failed = true;
            YR_LOGW("record alpha blend failed (%s), drawing labels onto the picture", rga_error(blended));
        }
    }
    if (impl_->blend_failed) {
        cpu_write_begin(impl_->video);
        OsdCanvas picture;
        picture.pixels = static_cast<std::uint8_t*>(impl_->video.ptr);
        picture.width = width;
        picture.height = height;
        picture.stride_bytes = impl_->hor * 4;
        osd_draw(picture, sample.result.detections);
        cpu_write_end(impl_->video);
    }

    {
        RgaApiLock lock;
        converted = convert_fd(impl_->video.fd, width, height, impl_->hor, impl_->ver, RK_FORMAT_BGRA_8888,
                               impl_->nv12.fd, width, height, impl_->hor, impl_->ver, RK_FORMAT_YCbCr_420_SP);
    }
    if (!rga_ok(converted)) {
        if (!impl_->rga_warned) {
            YR_LOGW("record RGA BGRA->NV12 failed: %s", rga_error(converted));
            impl_->rga_warned = true;
        }
        return Status::Ok();
    }

    mpp_frame_set_eos(impl_->enc_frame, 0);
    MppPacket packet = nullptr;
    MPP_RET ret = impl_->mpi->encode_put_frame(impl_->ctx, impl_->enc_frame);
    if (ret != MPP_OK) {
        return Status::Fail("mpp encode put failed: " + std::to_string(ret));
    }
    ret = impl_->mpi->encode_get_packet(impl_->ctx, &packet);
    if (ret != MPP_OK) {
        if (packet) {
            mpp_packet_deinit(&packet);
        }
        return Status::Fail("mpp encode get failed: " + std::to_string(ret));
    }
    if (packet == nullptr && !impl_->empty_warned) {
        impl_->empty_warned = true;
        YR_LOGW("record encode produced no packet");
    }
    std::int64_t pts = sample.frame.pts_us;
    if (pts < 0) {
        pts = static_cast<std::int64_t>(impl_->frames) * (1000000 / kFps);
    }
    return write_packet(packet, pts);
}

Status Mp4Recorder::write_packet(void* packet_ptr, std::int64_t pts_us) {
    MppPacket packet = packet_ptr;
    if (packet == nullptr || impl_->stream == nullptr) {
        if (packet) {
            mpp_packet_deinit(&packet);
        }
        return Status::Ok();
    }
    const std::uint8_t* payload = static_cast<const std::uint8_t*>(mpp_packet_get_pos(packet));
    const int payload_size = static_cast<int>(mpp_packet_get_length(packet));
    std::vector<std::uint8_t> avcc;
    bool keyframe = false;
    const bool has_slice = packet_avcc(payload, payload_size, &avcc, &keyframe);
    if (!has_slice && !impl_->empty_warned) {
        impl_->empty_warned = true;
        YR_LOGW("record packet has no slice, bytes=%d", payload_size);
    }
    mpp_packet_deinit(&packet);
    if (!has_slice) {
        return Status::Ok();
    }

    std::int64_t pts = pts_us;
    if (pts < 0) {
        pts = static_cast<std::int64_t>(impl_->frames) * (1000000 / kFps);
    }
    if (impl_->origin_pts < 0) {
        impl_->origin_pts = pts;
    }
    pts -= impl_->origin_pts;
    if (impl_->frames > 0 && pts <= impl_->last_pts) {
        pts = impl_->last_pts + 1000000 / kFps;
    }
    impl_->last_pts = pts;

    AVPacket* pkt = av_packet_alloc();
    if (pkt == nullptr) {
        return Status::Fail("av_packet_alloc failed");
    }
    if (av_new_packet(pkt, static_cast<int>(avcc.size())) < 0) {
        av_packet_free(&pkt);
        return Status::Fail("av_new_packet failed");
    }
    std::memcpy(pkt->data, avcc.data(), avcc.size());
    pkt->stream_index = impl_->stream->index;
    pkt->pts = av_rescale_q(pts, av_make_q(1, 1000000), impl_->stream->time_base);
    pkt->dts = pkt->pts;
    if (keyframe) {
        pkt->flags |= AV_PKT_FLAG_KEY;
    }
    const int averr = av_interleaved_write_frame(impl_->oc, pkt);
    av_packet_free(&pkt);
    if (averr < 0) {
        return Status::Fail("mp4 write failed: " + ffmpeg_err(averr));
    }
    ++impl_->frames;
    return Status::Ok();
}

}  // namespace yolo_rknn
