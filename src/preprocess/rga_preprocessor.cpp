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

}  // namespace

struct BufferCandidate {
    MppBufferType type;
    const char* name;
    bool cacheable;
};

struct RgbSlot {
    MppBufferGroup group = nullptr;
    MppBuffer buffer = nullptr;
    int fd = -1;
    void* ptr = nullptr;
    std::size_t bytes = 0;
    bool cacheable = false;
    bool sync_held = false;
};

void release_slot(RgbSlot* slot) {
    if (slot == nullptr) {
        return;
    }
    if (slot->sync_held && slot->fd >= 0) {
        dma_buf_end_cpu_read(slot->fd);
        slot->sync_held = false;
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
}

Status alloc_rgb_slot(RgbSlot* slot, std::size_t bytes, const char** allocator_name) {
    const BufferCandidate candidates[] = {
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG), "DRM CMA", false},
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CONTIG), "dma-heap CMA", false},
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG | MPP_BUFFER_FLAGS_CACHABLE),
         "DRM CMA cacheable", true},
        {static_cast<MppBufferType>(MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CONTIG | MPP_BUFFER_FLAGS_CACHABLE),
         "dma-heap CMA cacheable", true},
    };
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
        slot->cacheable = candidates[i].cacheable;
        if (allocator_name != nullptr) {
            *allocator_name = candidates[i].name;
        }
        return Status::Ok();
    }
    return Status::Fail("RGA destination contiguous buffer alloc failed");
}

IM_STATUS blit_window(int src_fd, int src_w, int src_h, int hor_stride, int ver_stride, int dst_fd, int dst_w,
                      int dst_h, const ModelWindow& window) {
    rga_buffer_t dst = wrapbuffer_fd_t(dst_fd, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_RGB_888);
    IM_STATUS filled = IM_STATUS_SUCCESS;
    if (window.fill) {
        im_rect full;
        full.x = 0;
        full.y = 0;
        full.width = dst_w;
        full.height = dst_h;
        filled = imfill(dst, full, 0x727272);
        if (!rga_ok(filled)) {
            return filled;
        }
    }
    rga_buffer_t src = wrapbuffer_fd_t(src_fd, src_w, src_h, hor_stride, ver_stride, RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t pat;
    std::memset(&pat, 0, sizeof(pat));
    im_rect src_rect;
    src_rect.x = window.src_x;
    src_rect.y = window.src_y;
    src_rect.width = window.src_w;
    src_rect.height = window.src_h;
    im_rect dst_rect;
    dst_rect.x = window.dst_x;
    dst_rect.y = window.dst_y;
    dst_rect.width = window.dst_w;
    dst_rect.height = window.dst_h;
    im_rect pat_rect;
    std::memset(&pat_rect, 0, sizeof(pat_rect));
    return improcess(src, dst, pat, src_rect, dst_rect, pat_rect, -1, nullptr, nullptr, IM_SYNC);
}

struct RgaPreprocessor::Impl {
    RgbSlot slot[2];
    int nslot = 1;
    bool parallel = false;
    int logged_src_w = 0;
    int logged_src_h = 0;
    bool logged_same = false;
};

RgaPreprocessor::RgaPreprocessor(const AppConfig& config) : config_(config), impl_(new Impl) {}

RgaPreprocessor::~RgaPreprocessor() {
    if (impl_ == nullptr) {
        return;
    }
    finish_cpu_read();
    release_slot(&impl_->slot[0]);
    release_slot(&impl_->slot[1]);
    delete impl_;
    impl_ = nullptr;
}

Status RgaPreprocessor::init() {
    ensure_rga();
    const std::size_t bytes =
        static_cast<std::size_t>(config_.model_width) * static_cast<std::size_t>(config_.model_height) * 3;
    impl_->parallel = config_.batch_size == 2 && config_.preprocess == "short_side";
    impl_->nslot = impl_->parallel ? 2 : 1;
    if (config_.batch_size == 2 && config_.preprocess != "short_side") {
        YR_LOGW("batch_size=2 is ignored unless preprocess=short_side; staying on one tile");
    }

    for (int i = 0; i < impl_->nslot; ++i) {
        const char* name = "";
        Status st = alloc_rgb_slot(&impl_->slot[i], bytes, &name);
        if (!st.ok()) {
            release_slot(&impl_->slot[0]);
            release_slot(&impl_->slot[1]);
            return st;
        }
        if (impl_->slot[i].cacheable) {
            YR_LOGW("RGB buffer %d is cacheable; NPU import still works after a dma-buf cache invalidate", i);
        }
        YR_LOGI("RGA dst RGB %dx%d tile=%d fd=%d bytes=%zu allocator=%s", config_.model_width, config_.model_height,
                i, impl_->slot[i].fd, bytes, name);
    }
    return Status::Ok();
}

void RgaPreprocessor::finish_cpu_read() {
    if (impl_ == nullptr) {
        return;
    }
    for (int i = 0; i < 2; ++i) {
        if (!impl_->slot[i].sync_held) {
            continue;
        }
        dma_buf_end_cpu_read(impl_->slot[i].fd);
        impl_->slot[i].sync_held = false;
    }
}

Status RgaPreprocessor::process(const DecodedFrame& frame, PreprocessResult* tiles, int* tile_count) {
    finish_cpu_read();
    if (tiles == nullptr || tile_count == nullptr || *tile_count < 1) {
        return Status::Fail("preprocess output is missing");
    }
    if (impl_->slot[0].fd < 0 || impl_->slot[0].ptr == nullptr) {
        return Status::Fail("RGA preprocessor is not initialized");
    }
    if (frame.fd < 0 || frame.width < 2 || frame.height < 2) {
        return Status::Fail("decoded frame has no NV12 dma-buf");
    }

    const int src_w = frame.width & ~1;
    const int src_h = frame.height & ~1;
    const int dst_w = config_.model_width;
    const int dst_h = config_.model_height;
    const InputFit fit = config_.preprocess == "short_side" ? InputFit::ShortSide : InputFit::Letterbox;
    ModelWindow windows[2];
    int n = 1;
    if (impl_->parallel) {
        if (!plan_model_window(InputFit::ShortSide, src_w, src_h, dst_w, dst_h, &windows[0], WindowAnchor::Start) ||
            !plan_model_window(InputFit::ShortSide, src_w, src_h, dst_w, dst_h, &windows[1], WindowAnchor::End)) {
            return Status::Fail("preprocess window is empty");
        }
        const bool same = windows[0].src_x == windows[1].src_x && windows[0].src_y == windows[1].src_y &&
                          windows[0].src_w == windows[1].src_w && windows[0].src_h == windows[1].src_h;
        if (same) {
            n = 1;
            if (!impl_->logged_same) {
                YR_LOGI("short_side edges cover the same window, batch stays 1");
                impl_->logged_same = true;
            }
        } else {
            n = 2;
        }
    } else if (!plan_model_window(fit, src_w, src_h, dst_w, dst_h, &windows[0])) {
        return Status::Fail("preprocess window is empty");
    }
    if (n > *tile_count || n > impl_->nslot) {
        return Status::Fail("preprocess tile capacity is too small");
    }
    if (impl_->logged_src_w != src_w || impl_->logged_src_h != src_h) {
        impl_->logged_src_w = src_w;
        impl_->logged_src_h = src_h;
        for (int i = 0; i < n; ++i) {
            YR_LOGI("preprocess %s src %dx%d tile=%d window %d,%d %dx%d -> %d,%d %dx%d", config_.preprocess.c_str(),
                    src_w, src_h, i, windows[i].src_x, windows[i].src_y, windows[i].src_w, windows[i].src_h,
                    windows[i].dst_x, windows[i].dst_y, windows[i].dst_w, windows[i].dst_h);
        }
    }

    IM_STATUS blitted = IM_STATUS_SUCCESS;
    {
        // 和显示线程的 OSD 清屏共用一把锁，librga 不能并发 ioctl。两张切片在这把锁里串行缩放。
        RgaApiLock lock;
        for (int i = 0; i < n; ++i) {
            blitted = blit_window(frame.fd, src_w, src_h, frame.hor_stride, frame.ver_stride, impl_->slot[i].fd,
                                  dst_w, dst_h, windows[i]);
            if (!rga_ok(blitted)) {
                break;
            }
        }
    }
    if (!rga_ok(blitted)) {
        return Status::Fail(std::string("RGA blit NV12->RGB failed: ") + rga_error(blitted));
    }

    for (int i = 0; i < n; ++i) {
        if (impl_->slot[i].cacheable) {
            if (!dma_buf_begin_cpu_read(impl_->slot[i].fd) && !g_sync_warned) {
                YR_LOGW("dma-buf CPU read sync failed; continuing in case the mapping is coherent");
                g_sync_warned = true;
            }
            impl_->slot[i].sync_held = true;
        }
        tiles[i] = PreprocessResult();
        tiles[i].rgb = static_cast<const std::uint8_t*>(impl_->slot[i].ptr);
        tiles[i].bytes = impl_->slot[i].bytes;
        tiles[i].fd = impl_->slot[i].fd;
        tiles[i].meta = windows[i].meta;
    }
    *tile_count = n;
    return Status::Ok();
}

int RgaPreprocessor::tile_count() const { return impl_ != nullptr ? impl_->nslot : 0; }

int RgaPreprocessor::buffer_fd(int tile) const {
    if (impl_ == nullptr || tile < 0 || tile >= impl_->nslot) {
        return -1;
    }
    return impl_->slot[tile].fd;
}

void* RgaPreprocessor::buffer_ptr(int tile) const {
    if (impl_ == nullptr || tile < 0 || tile >= impl_->nslot) {
        return nullptr;
    }
    return impl_->slot[tile].ptr;
}

std::size_t RgaPreprocessor::buffer_bytes(int tile) const {
    if (impl_ == nullptr || tile < 0 || tile >= impl_->nslot) {
        return 0;
    }
    return impl_->slot[tile].bytes;
}

}  // namespace yolo_rknn
