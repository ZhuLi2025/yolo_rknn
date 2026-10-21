#define MODULE_TAG "yr_rga"

#include "yolo_rknn/preprocess/rga_preprocessor.hpp"

#include "yolo_rknn/common/dma_buf.hpp"
#include "yolo_rknn/common/logger.hpp"
#include "yolo_rknn/decode/mpp_c_api.hpp"
#include "yolo_rknn/preprocess/rga_lock.hpp"

#include "im2d.h"

#include <cstring>
#include <mutex>
#include <string>

namespace yolo_rknn {
namespace {

std::once_flag g_rga_once;
bool g_sync_warned = false;

bool rga_ok(IM_STATUS status) { return status == IM_STATUS_SUCCESS || status == IM_STATUS_NOERROR; }

const char* rga_error(IM_STATUS status) {
    const char* text = imStrError_t(status);
    return text != nullptr ? text : "unknown";
}

void ensure_rga() {
    std::call_once(g_rga_once, [] {
        const char* version = querystring(RGA_VERSION);
        if (version == nullptr) {
            YR_LOGI("RGA unknown");
            return;
        }
        std::string text(version);
        for (std::size_t i = 0; i < text.size(); ++i) {
            if (text[i] == '\n' || text[i] == '\r') {
                text[i] = ' ';
            }
        }
        YR_LOGI("RGA %s", text.c_str());
    });
}

struct BufferCandidate {
    MppBufferType type;
    const char* name;
    bool cacheable;
};

}  // namespace

struct RgaPreprocessor::Impl {
    MppBufferGroup group = nullptr;
    MppBuffer buffer = nullptr;
    int fd = -1;
    void* ptr = nullptr;
    std::size_t bytes = 0;
    bool cacheable = false;
    bool sync_held = false;
};

RgaPreprocessor::RgaPreprocessor(const AppConfig& config) : config_(config), impl_(new Impl) {}

RgaPreprocessor::~RgaPreprocessor() {
    if (impl_ == nullptr) {
        return;
    }
    finish_cpu_read();
    if (impl_->buffer) {
        mpp_buffer_put(impl_->buffer);
        impl_->buffer = nullptr;
    }
    if (impl_->group) {
        mpp_buffer_group_put(impl_->group);
        impl_->group = nullptr;
    }
    delete impl_;
    impl_ = nullptr;
}

Status RgaPreprocessor::init() {
    ensure_rga();
    const std::size_t bytes =
        static_cast<std::size_t>(config_.model_width) * static_cast<std::size_t>(config_.model_height) * 3;

    // 非 cacheable 的物理连续内存在 RGA 写完后 NPU 可以直接读。
    // 当前用户打得开 DRM，打不开 /dev/dma_heap（那个节点只给 video 组）。所以 DRM 放前面。
    const BufferCandidate candidates[] = {
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG), "DRM CMA", false},
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CONTIG), "dma-heap CMA", false},
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG | MPP_BUFFER_FLAGS_CACHABLE),
         "DRM CMA cacheable", true},
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CONTIG | MPP_BUFFER_FLAGS_CACHABLE),
         "dma-heap CMA cacheable", true},
    };

    MppBufferGroup group = nullptr;
    MppBuffer buffer = nullptr;
    const BufferCandidate* chosen = nullptr;
    for (std::size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        MppBufferGroup trial = nullptr;
        MPP_RET ret = mpp_buffer_group_get_internal(&trial, candidates[i].type);
        if (ret != MPP_OK || trial == nullptr) {
            continue;
        }
        MppBuffer trial_buf = nullptr;
        ret = mpp_buffer_get(trial, &trial_buf, bytes);
        if (ret != MPP_OK || trial_buf == nullptr) {
            mpp_buffer_group_put(trial);
            continue;
        }
        group = trial;
        buffer = trial_buf;
        chosen = &candidates[i];
        break;
    }
    if (group == nullptr || buffer == nullptr || chosen == nullptr) {
        return Status::Fail("RGA destination contiguous buffer alloc failed");
    }
    if (chosen->cacheable) {
        YR_LOGW("RGB buffer is cacheable; NPU import still works after a dma-buf cache invalidate");
    }

    void* ptr = mpp_buffer_get_ptr(buffer);
    const int fd = mpp_buffer_get_fd(buffer);
    if (ptr == nullptr || fd < 0 || mpp_buffer_get_size(buffer) < bytes) {
        mpp_buffer_put(buffer);
        mpp_buffer_group_put(group);
        return Status::Fail("RGA destination buffer has no cpu pointer or dma-buf fd");
    }
    impl_->group = group;
    impl_->buffer = buffer;
    impl_->ptr = ptr;
    impl_->fd = fd;
    impl_->bytes = bytes;
    impl_->cacheable = chosen->cacheable;
    YR_LOGI("RGA dst RGB %dx%d fd=%d bytes=%zu allocator=%s", config_.model_width, config_.model_height, fd, bytes,
            chosen->name);
    return Status::Ok();
}

void RgaPreprocessor::finish_cpu_read() {
    if (impl_ == nullptr || !impl_->sync_held) {
        return;
    }
    dma_buf_end_cpu_read(impl_->fd);
    impl_->sync_held = false;
}

Status RgaPreprocessor::process(const DecodedFrame& frame, PreprocessResult& out) {
    finish_cpu_read();
    out = PreprocessResult();
    if (impl_->fd < 0 || impl_->ptr == nullptr) {
        return Status::Fail("RGA preprocessor is not initialized");
    }
    if (frame.fd < 0 || frame.width < 2 || frame.height < 2) {
        return Status::Fail("decoded frame has no NV12 dma-buf");
    }

    const int src_w = frame.width & ~1;
    const int src_h = frame.height & ~1;
    const int dst_w = config_.model_width;
    const int dst_h = config_.model_height;

    // letterbox：按短边缩放，空出来的边由 RGA 填灰，而不是 CPU memset 一整张图。
    const float scale_w = static_cast<float>(dst_w) / static_cast<float>(src_w);
    const float scale_h = static_cast<float>(dst_h) / static_cast<float>(src_h);
    const float scale = scale_w < scale_h ? scale_w : scale_h;
    int new_w = static_cast<int>(static_cast<float>(src_w) * scale) & ~1;
    int new_h = static_cast<int>(static_cast<float>(src_h) * scale) & ~1;
    if (new_w < 2) {
        new_w = 2;
    }
    if (new_h < 2) {
        new_h = 2;
    }
    if (new_w > dst_w) {
        new_w = dst_w;
    }
    if (new_h > dst_h) {
        new_h = dst_h;
    }
    const int pad_x = ((dst_w - new_w) / 2) & ~1;
    const int pad_y = ((dst_h - new_h) / 2) & ~1;

    rga_buffer_t dst = wrapbuffer_fd_t(impl_->fd, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_RGB_888);
    im_rect full;
    full.x = 0;
    full.y = 0;
    full.width = dst_w;
    full.height = dst_h;
    IM_STATUS filled = IM_STATUS_FAILED;
    IM_STATUS blitted = IM_STATUS_FAILED;
    {
        // 和显示线程的 OSD 清屏共用一把锁，librga 不能并发 ioctl。
        RgaApiLock lock;
        filled = imfill(dst, full, 0x727272);
        if (rga_ok(filled)) {
            rga_buffer_t src =
                wrapbuffer_fd_t(frame.fd, src_w, src_h, frame.hor_stride, frame.ver_stride, RK_FORMAT_YCbCr_420_SP);
            rga_buffer_t pat;
            std::memset(&pat, 0, sizeof(pat));
            im_rect src_rect;
            src_rect.x = 0;
            src_rect.y = 0;
            src_rect.width = src_w;
            src_rect.height = src_h;
            im_rect dst_rect;
            dst_rect.x = pad_x;
            dst_rect.y = pad_y;
            dst_rect.width = new_w;
            dst_rect.height = new_h;
            im_rect pat_rect;
            std::memset(&pat_rect, 0, sizeof(pat_rect));
            // librga 按矩形把这一帧调度到 RGA2 或 RGA3。IM_SYNC 表示 ioctl 返回时缩放已经写完。
            blitted = improcess(src, dst, pat, src_rect, dst_rect, pat_rect, -1, nullptr, nullptr, IM_SYNC);
        }
    }
    if (!rga_ok(filled)) {
        return Status::Fail(std::string("RGA color fill failed: ") + rga_error(filled));
    }
    if (!rga_ok(blitted)) {
        return Status::Fail(std::string("RGA blit NV12->RGB failed: ") + rga_error(blitted));
    }

    if (impl_->cacheable) {
        if (!dma_buf_begin_cpu_read(impl_->fd) && !g_sync_warned) {
            YR_LOGW("dma-buf CPU read sync failed; continuing in case the mapping is coherent");
            g_sync_warned = true;
        }
        impl_->sync_held = true;
    }

    out.rgb = static_cast<const std::uint8_t*>(impl_->ptr);
    out.bytes = impl_->bytes;
    out.fd = impl_->fd;
    out.meta.scale = static_cast<float>(new_w) / static_cast<float>(src_w);
    out.meta.pad_x = pad_x;
    out.meta.pad_y = pad_y;
    out.meta.src_width = src_w;
    out.meta.src_height = src_h;
    out.meta.model_width = dst_w;
    out.meta.model_height = dst_h;
    return Status::Ok();
}

int RgaPreprocessor::buffer_fd() const { return impl_ != nullptr ? impl_->fd : -1; }

void* RgaPreprocessor::buffer_ptr() const { return impl_ != nullptr ? impl_->ptr : nullptr; }

std::size_t RgaPreprocessor::buffer_bytes() const { return impl_ != nullptr ? impl_->bytes : 0; }

}  // namespace yolo_rknn
