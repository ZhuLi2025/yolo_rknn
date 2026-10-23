#define MODULE_TAG "yr_helmet"

#include "yolo_rknn/infer/helmet_classifier.hpp"

#include "yolo_rknn/common/dma_buf.hpp"
#include "yolo_rknn/common/logger.hpp"
#include "yolo_rknn/decode/mpp_c_api.hpp"
#include "yolo_rknn/preprocess/rga_lock.hpp"

#include <rknn_api.h>

#include "im2d.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace yolo_rknn {
namespace {

bool rga_ok(IM_STATUS status) { return status == IM_STATUS_SUCCESS || status == IM_STATUS_NOERROR; }

const char* rga_error(IM_STATUS status) {
    const char* text = imStrError_t(status);
    return text != nullptr ? text : "unknown";
}

struct RgbSlot {
    MppBufferGroup group = nullptr;
    MppBuffer buffer = nullptr;
    int fd = -1;
    void* ptr = nullptr;
    std::size_t bytes = 0;
    bool cacheable = false;
};

void release_slot(RgbSlot* slot) {
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
}

Status alloc_slot(RgbSlot* slot, std::size_t bytes) {
    const MppBufferType types[] = {
        static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG),
        static_cast<MppBufferType>(MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CONTIG),
        static_cast<MppBufferType>(MPP_BUFFER_TYPE_DRM | MPP_BUFFER_FLAGS_CONTIG | MPP_BUFFER_FLAGS_CACHABLE),
        static_cast<MppBufferType>(MPP_BUFFER_TYPE_DMA_HEAP | MPP_BUFFER_FLAGS_CONTIG | MPP_BUFFER_FLAGS_CACHABLE),
    };
    const bool cacheable[] = {false, false, true, true};
    for (std::size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
        MppBufferGroup group = nullptr;
        if (mpp_buffer_group_get_internal(&group, types[i]) != MPP_OK || group == nullptr) {
            continue;
        }
        MppBuffer buffer = nullptr;
        if (mpp_buffer_get(group, &buffer, bytes) != MPP_OK || buffer == nullptr) {
            mpp_buffer_group_put(group);
            continue;
        }
        void* ptr = mpp_buffer_get_ptr(buffer);
        const int fd = mpp_buffer_get_fd(buffer);
        if (ptr == nullptr || fd < 0) {
            mpp_buffer_put(buffer);
            mpp_buffer_group_put(group);
            continue;
        }
        slot->group = group;
        slot->buffer = buffer;
        slot->ptr = ptr;
        slot->fd = fd;
        slot->bytes = bytes;
        slot->cacheable = cacheable[i];
        return Status::Ok();
    }
    return Status::Fail("helmet RGB buffer alloc failed");
}

int even_floor(int value) { return value & ~1; }

}  // namespace

struct HelmetClassifier::Impl {
    rknn_context ctx = 0;
    rknn_tensor_mem* input_mem = nullptr;
    bool zero_copy = false;
    int width = 0;
    int height = 0;
    std::vector<float> scores;
    RgbSlot rgb;
    bool crop_warned = false;
    bool ran_logged = false;
};

HelmetClassifier::HelmetClassifier(const AppConfig& config) : config_(config), impl_(new Impl) {}

HelmetClassifier::~HelmetClassifier() {
    if (impl_ == nullptr) {
        return;
    }
    if (impl_->ctx) {
        if (impl_->input_mem) {
            rknn_destroy_mem(impl_->ctx, impl_->input_mem);
        }
        rknn_destroy(impl_->ctx);
    }
    release_slot(&impl_->rgb);
    delete impl_;
    impl_ = nullptr;
}

Status HelmetClassifier::init() {
    std::ifstream in(config_.helmet_model.c_str(), std::ios::binary);
    if (!in) {
        return Status::Fail("cannot open helmet model: " + config_.helmet_model);
    }
    std::vector<unsigned char> model((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (model.empty()) {
        return Status::Fail("helmet model is empty: " + config_.helmet_model);
    }
    const int ret = rknn_init(&impl_->ctx, model.data(), static_cast<uint32_t>(model.size()), RKNN_FLAG_PRIOR_HIGH,
                              nullptr);
    if (ret != RKNN_SUCC) {
        return Status::Fail("helmet rknn_init failed: " + std::to_string(ret));
    }
    const int core_ret = rknn_set_core_mask(impl_->ctx, RKNN_NPU_CORE_2);
    if (core_ret != RKNN_SUCC) {
        return Status::Fail("helmet rknn_set_core_mask core2 failed: " + std::to_string(core_ret));
    }

    rknn_input_output_num io;
    std::memset(&io, 0, sizeof(io));
    if (rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != RKNN_SUCC || io.n_input != 1 ||
        io.n_output != 1) {
        return Status::Fail("helmet model must have 1 input and 1 output");
    }
    rknn_tensor_attr input;
    std::memset(&input, 0, sizeof(input));
    input.index = 0;
    if (rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR, &input, sizeof(input)) != RKNN_SUCC || input.n_dims != 4 ||
        input.fmt != RKNN_TENSOR_NHWC || input.dims[3] != 3) {
        return Status::Fail("helmet input must be NHWC RGB");
    }
    impl_->width = static_cast<int>(input.dims[2]);
    impl_->height = static_cast<int>(input.dims[1]);
    if (impl_->width < 16 || impl_->height < 16 || (impl_->width % 16) != 0 || (impl_->height % 16) != 0) {
        return Status::Fail("helmet input width and height must be multiples of 16");
    }

    rknn_tensor_attr output;
    std::memset(&output, 0, sizeof(output));
    output.index = 0;
    if (rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &output, sizeof(output)) != RKNN_SUCC || output.n_elems != 2) {
        return Status::Fail("helmet output must be 2 scores, channel 0=helmet and channel 1=no helmet");
    }
    impl_->scores.assign(2, 0.f);

    const std::size_t bytes = static_cast<std::size_t>(impl_->width) * static_cast<std::size_t>(impl_->height) * 3u;
    Status st = alloc_slot(&impl_->rgb, bytes);
    if (!st.ok()) {
        return st;
    }
    rknn_tensor_mem* mem =
        rknn_create_mem_from_fd(impl_->ctx, impl_->rgb.fd, impl_->rgb.ptr, static_cast<uint32_t>(bytes), 0);
    if (mem != nullptr) {
        rknn_tensor_attr attr = input;
        attr.pass_through = 0;
        attr.type = RKNN_TENSOR_UINT8;
        attr.fmt = RKNN_TENSOR_NHWC;
        if (rknn_set_io_mem(impl_->ctx, mem, &attr) == RKNN_SUCC) {
            impl_->input_mem = mem;
            impl_->zero_copy = true;
        } else {
            rknn_destroy_mem(impl_->ctx, mem);
        }
    }
    YR_LOGI("helmet classifier %dx%d core=2 zero_copy=%d classes=helmet,no helmet", impl_->width, impl_->height,
            impl_->zero_copy ? 1 : 0);
    return Status::Ok();
}

Status HelmetClassifier::classify(const DecodedFrame& frame, std::vector<Detection>& detections) {
    if (impl_->ctx == 0 || frame.fd < 0) {
        return Status::Fail("helmet classifier has no frame");
    }
    const int src_w = even_floor(frame.width);
    const int src_h = even_floor(frame.height);
    if (src_w < 2 || src_h < 2) {
        return Status::Fail("helmet source frame is empty");
    }

    for (std::size_t i = 0; i < detections.size(); ++i) {
        Detection& det = detections[i];
        det.helmet = -1;
        det.helmet_score = 0.f;
        int x = even_floor(static_cast<int>(det.x));
        int y = even_floor(static_cast<int>(det.y));
        int w = even_floor(static_cast<int>(det.w));
        int h = even_floor(static_cast<int>(det.h));
        if (x < 0) {
            x = 0;
        }
        if (y < 0) {
            y = 0;
        }
        if (w < 2) {
            w = 2;
        }
        if (h < 2) {
            h = 2;
        }
        if (x + w > src_w) {
            w = even_floor(src_w - x);
        }
        if (y + h > src_h) {
            h = even_floor(src_h - y);
        }
        if (w < 2 || h < 2) {
            continue;
        }

        IM_STATUS blitted = IM_STATUS_FAILED;
        {
            RgaApiLock lock;
            rga_buffer_t src =
                wrapbuffer_fd_t(frame.fd, src_w, src_h, frame.hor_stride, frame.ver_stride, RK_FORMAT_YCbCr_420_SP);
            rga_buffer_t dst =
                wrapbuffer_fd_t(impl_->rgb.fd, impl_->width, impl_->height, impl_->width, impl_->height, RK_FORMAT_RGB_888);
            rga_buffer_t pat;
            std::memset(&pat, 0, sizeof(pat));
            im_rect src_rect;
            src_rect.x = x;
            src_rect.y = y;
            src_rect.width = w;
            src_rect.height = h;
            im_rect dst_rect;
            dst_rect.x = 0;
            dst_rect.y = 0;
            dst_rect.width = impl_->width;
            dst_rect.height = impl_->height;
            im_rect pat_rect;
            std::memset(&pat_rect, 0, sizeof(pat_rect));
            blitted = improcess(src, dst, pat, src_rect, dst_rect, pat_rect, -1, nullptr, nullptr, IM_SYNC);
        }
        if (!rga_ok(blitted)) {
            if (!impl_->crop_warned) {
                YR_LOGW("helmet crop failed: %s", rga_error(blitted));
                impl_->crop_warned = true;
            }
            continue;
        }
        if (impl_->rgb.cacheable) {
            dma_buf_begin_cpu_read(impl_->rgb.fd);
        }

        if (!impl_->zero_copy) {
            rknn_input feed;
            std::memset(&feed, 0, sizeof(feed));
            feed.index = 0;
            feed.buf = impl_->rgb.ptr;
            feed.size = static_cast<uint32_t>(impl_->rgb.bytes);
            feed.pass_through = 0;
            feed.type = RKNN_TENSOR_UINT8;
            feed.fmt = RKNN_TENSOR_NHWC;
            if (rknn_inputs_set(impl_->ctx, 1, &feed) != RKNN_SUCC) {
                return Status::Fail("helmet rknn_inputs_set failed");
            }
        }
        if (rknn_run(impl_->ctx, nullptr) != RKNN_SUCC) {
            return Status::Fail("helmet rknn_run failed");
        }
        rknn_output fetched;
        std::memset(&fetched, 0, sizeof(fetched));
        fetched.want_float = 1;
        fetched.is_prealloc = 1;
        fetched.index = 0;
        fetched.buf = impl_->scores.data();
        fetched.size = static_cast<uint32_t>(impl_->scores.size() * sizeof(float));
        if (rknn_outputs_get(impl_->ctx, 1, &fetched, nullptr) != RKNN_SUCC) {
            return Status::Fail("helmet rknn_outputs_get failed");
        }
        rknn_outputs_release(impl_->ctx, 1, &fetched);
        if (impl_->rgb.cacheable) {
            dma_buf_end_cpu_read(impl_->rgb.fd);
        }

        const float max_logit = std::max(impl_->scores[0], impl_->scores[1]);
        const float p0 = std::exp(impl_->scores[0] - max_logit);
        const float p1 = std::exp(impl_->scores[1] - max_logit);
        const float sum = p0 + p1;
        if (p0 >= p1) {
            det.helmet = 0;
            det.helmet_score = sum > 0.f ? p0 / sum : 0.f;
        } else {
            det.helmet = 1;
            det.helmet_score = sum > 0.f ? p1 / sum : 0.f;
        }
    }
    if (!impl_->ran_logged && !detections.empty()) {
        YR_LOGI("helmet classified %zu boxes", detections.size());
        impl_->ran_logged = true;
    }
    return Status::Ok();
}

}  // namespace yolo_rknn
