#define MODULE_TAG "yr_rga"

#include "yolo_rknn/preprocess/rga_preprocessor.hpp"

#include "yolo_rknn/common/dma_buf.hpp"
#include "yolo_rknn/common/logger.hpp"
#include "yolo_rknn/decode/mpp_c_api.hpp"
#include "yolo_rknn/preprocess/rga_lock.hpp"

extern "C" {
#include "RgaApi.h"
#include "rga.h"
}

#include <cstring>
#include <mutex>

namespace yolo_rknn {
namespace {

std::once_flag g_rga_once;
bool g_sync_warned = false;

void ensure_rga() {
    std::call_once(g_rga_once, [] {
        const int ret = c_RkRgaInit();
        if (ret != 0) {
            YR_LOGW("c_RkRgaInit -> %d", ret);
        }
    });
}

}  // namespace

struct RgaPreprocessor::Impl {
    MppBufferGroup group = nullptr;
    MppBuffer buffer = nullptr;
    int fd = -1;
    void* ptr = nullptr;
    std::size_t bytes = 0;
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

    // 模型输入只有 640x640 量级，CPU/RKNN 要读它，所以尽量要 cacheable。
    // 读之前会做 dma-buf sync，把 RGA 的 DMA 写入从 cache 里作废。
    MppBufferType cached = static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG |
                                                      MPP_BUFFER_FLAGS_CACHABLE);
    MppBufferGroup group = nullptr;
    MPP_RET ret = mpp_buffer_group_get_internal(&group, cached);
    if (ret != MPP_OK || group == nullptr) {
        const MppBufferType plain =
            static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG);
        ret = mpp_buffer_group_get_internal(&group, plain);
        if (ret != MPP_OK || group == nullptr) {
            return Status::Fail("RGA destination DRM buffer group alloc failed");
        }
        YR_LOGW("RGB buffer is not cacheable; RKNN will read it uncached");
    }

    MppBuffer buffer = nullptr;
    ret = mpp_buffer_get(group, &buffer, bytes);
    if (ret != MPP_OK || buffer == nullptr) {
        mpp_buffer_group_put(group);
        return Status::Fail("RGA destination buffer alloc failed");
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
    YR_LOGI("RGA dst RGB %dx%d fd=%d bytes=%zu", config_.model_width, config_.model_height, fd, bytes);
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

    rga_info_t dst;
    std::memset(&dst, 0, sizeof(dst));
    dst.fd = impl_->fd;
    dst.mmuFlag = 1;
    dst.format = RK_FORMAT_RGB_888;
    // 114 灰，和 Ultralytics letterbox 的填充值一致。打包格式按 R|G<<8|B<<16。
    dst.color = 0x00727272;
    rga_set_rect(&dst.rect, 0, 0, dst_w, dst_h, dst_w, dst_h, RK_FORMAT_RGB_888);
    int filled = 0;
    int blitted = 0;
    {
        // 和显示线程的 OSD 清屏共用一把锁，librga 不能并发 ioctl。
        RgaApiLock lock;
        filled = c_RkRgaColorFill(&dst);
        if (filled == 0) {
            rga_info_t src;
            std::memset(&src, 0, sizeof(src));
            src.fd = frame.fd;
            src.mmuFlag = 1;
            src.format = RK_FORMAT_YCbCr_420_SP;
            rga_set_rect(&src.rect, 0, 0, src_w, src_h, frame.hor_stride, frame.ver_stride,
                         RK_FORMAT_YCbCr_420_SP);
            std::memset(&dst, 0, sizeof(dst));
            dst.fd = impl_->fd;
            dst.mmuFlag = 1;
            dst.format = RK_FORMAT_RGB_888;
            rga_set_rect(&dst.rect, pad_x, pad_y, new_w, new_h, dst_w, dst_h, RK_FORMAT_RGB_888);
            blitted = c_RkRgaBlit(&src, &dst, nullptr);
        }
    }
    if (filled != 0) {
        return Status::Fail("RGA color fill failed: " + std::to_string(filled));
    }

    if (blitted != 0) {
        // 失败就丢掉这一帧。不存在 CPU 缩放的后备路径。
        return Status::Fail("RGA blit NV12->RGB failed: " + std::to_string(blitted));
    }

    if (!dma_buf_begin_cpu_read(impl_->fd) && !g_sync_warned) {
        YR_LOGW("dma-buf CPU read sync failed; continuing in case the mapping is coherent");
        g_sync_warned = true;
    }
    impl_->sync_held = true;

    out.rgb = static_cast<const std::uint8_t*>(impl_->ptr);
    out.bytes = impl_->bytes;
    out.meta.scale = static_cast<float>(new_w) / static_cast<float>(src_w);
    out.meta.pad_x = pad_x;
    out.meta.pad_y = pad_y;
    out.meta.src_width = src_w;
    out.meta.src_height = src_h;
    out.meta.model_width = dst_w;
    out.meta.model_height = dst_h;
    return Status::Ok();
}

}  // namespace yolo_rknn
