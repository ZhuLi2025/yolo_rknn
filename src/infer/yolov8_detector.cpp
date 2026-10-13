#include "yolo_rknn/infer/yolov8_detector.hpp"

#include "yolo_rknn/common/logger.hpp"
#include "yolo_rknn/infer/postprocess.hpp"

#include <rockchip/rknn_api.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace yolo_rknn {
namespace {

std::string shape_string(const rknn_tensor_attr& attr) {
    std::ostringstream os;
    os << "[";
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        if (i) {
            os << ",";
        }
        os << attr.dims[i];
    }
    os << "] type=" << static_cast<int>(attr.type) << " fmt=" << static_cast<int>(attr.fmt)
       << " elems=" << attr.n_elems << " bytes=" << attr.size;
    return os.str();
}

bool input_matches(const rknn_tensor_attr& attr, int width, int height) {
    const uint32_t expect = static_cast<uint32_t>(width) * static_cast<uint32_t>(height) * 3u;
    if (attr.n_elems == expect) {
        return true;
    }
    if (attr.n_dims == 4 && attr.fmt == RKNN_TENSOR_NHWC && attr.dims[1] == static_cast<uint32_t>(height) &&
        attr.dims[2] == static_cast<uint32_t>(width) && attr.dims[3] == 3) {
        return true;
    }
    if (attr.n_dims == 4 && attr.fmt == RKNN_TENSOR_NCHW && attr.dims[1] == 3 &&
        attr.dims[2] == static_cast<uint32_t>(height) && attr.dims[3] == static_cast<uint32_t>(width)) {
        return true;
    }
    return false;
}

}  // namespace

struct YoloV8Detector::Impl {
    rknn_context ctx = 0;
    rknn_tensor_attr input;
    std::vector<rknn_tensor_attr> outputs;
    std::vector<std::vector<float>> host;
    bool pass_through = false;
    bool layout_warned = false;

    Impl() { std::memset(&input, 0, sizeof(input)); }
};

YoloV8Detector::YoloV8Detector(const AppConfig& config) : config_(config), impl_(new Impl) {}

YoloV8Detector::~YoloV8Detector() {
    if (impl_ == nullptr) {
        return;
    }
    if (impl_->ctx) {
        rknn_destroy(impl_->ctx);
        impl_->ctx = 0;
    }
    delete impl_;
    impl_ = nullptr;
}

Status YoloV8Detector::init() {
    std::ifstream in(config_.model_path.c_str(), std::ios::binary);
    if (!in) {
        return Status::Fail("cannot open rknn model: " + config_.model_path);
    }
    std::vector<unsigned char> model((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (model.empty()) {
        return Status::Fail("rknn model is empty: " + config_.model_path);
    }

    // 不用 RKNN_FLAG_ASYNC_MASK：那个模式取回的是上一帧的结果。
    const int ret = rknn_init(&impl_->ctx, model.data(), static_cast<uint32_t>(model.size()), RKNN_FLAG_PRIOR_HIGH);
    if (ret != RKNN_SUCC) {
        return Status::Fail("rknn_init failed: " + std::to_string(ret) +
                            " (model must be converted for RK3399Pro with RKNN toolkit 1.x)");
    }

    rknn_sdk_version ver;
    std::memset(&ver, 0, sizeof(ver));
    if (rknn_query(impl_->ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver)) == RKNN_SUCC) {
        YR_LOGI("rknn api=%s drv=%s", ver.api_version, ver.drv_version);
    }

    rknn_input_output_num io;
    std::memset(&io, 0, sizeof(io));
    if (rknn_query(impl_->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != RKNN_SUCC || io.n_input < 1 ||
        io.n_output < 1) {
        return Status::Fail("rknn_query input/output count failed");
    }

    impl_->input.index = 0;
    if (rknn_query(impl_->ctx, RKNN_QUERY_INPUT_ATTR, &impl_->input, sizeof(impl_->input)) != RKNN_SUCC) {
        return Status::Fail("rknn_query input attr failed");
    }
    YR_LOGI("rknn input0 %s", shape_string(impl_->input).c_str());
    if (!input_matches(impl_->input, config_.model_width, config_.model_height)) {
        return Status::Fail("model input shape does not match model_width x model_height x 3: " +
                            shape_string(impl_->input));
    }
    const uint32_t rgb_bytes =
        static_cast<uint32_t>(config_.model_width) * static_cast<uint32_t>(config_.model_height) * 3u;
    impl_->pass_through = impl_->input.type == RKNN_TENSOR_UINT8 && impl_->input.fmt == RKNN_TENSOR_NHWC &&
                          impl_->input.size == rgb_bytes;
    YR_LOGI("rknn input pass_through=%d", impl_->pass_through ? 1 : 0);

    impl_->outputs.resize(io.n_output);
    impl_->host.resize(io.n_output);
    for (uint32_t i = 0; i < io.n_output; ++i) {
        rknn_tensor_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.index = i;
        if (rknn_query(impl_->ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) != RKNN_SUCC) {
            return Status::Fail("rknn_query output attr failed");
        }
        YR_LOGI("rknn output%u %s", i, shape_string(attr).c_str());
        impl_->outputs[i] = attr;
        impl_->host[i].assign(attr.n_elems, 0.f);
    }
    return Status::Ok();
}

Status YoloV8Detector::infer(const PreprocessResult& input, InferenceResult& result) {
    if (impl_->ctx == 0) {
        return Status::Fail("detector is not initialized");
    }
    if (input.rgb == nullptr || input.bytes == 0) {
        return Status::Fail("empty preprocess buffer");
    }

    rknn_input feed;
    std::memset(&feed, 0, sizeof(feed));
    feed.index = 0;
    feed.buf = const_cast<std::uint8_t*>(input.rgb);
    feed.size = static_cast<uint32_t>(input.bytes);
    if (impl_->pass_through) {
        feed.pass_through = 1;
    } else {
        // 驱动把 RGB888 转成模型内部布局。这一步只发生在模型输入张量上。
        feed.pass_through = 0;
        feed.type = RKNN_TENSOR_UINT8;
        feed.fmt = RKNN_TENSOR_NHWC;
    }

    const auto t0 = std::chrono::steady_clock::now();
    int ret = rknn_inputs_set(impl_->ctx, 1, &feed);
    if (ret != RKNN_SUCC) {
        return Status::Fail("rknn_inputs_set failed: " + std::to_string(ret));
    }
    ret = rknn_run(impl_->ctx, nullptr);
    if (ret != RKNN_SUCC) {
        return Status::Fail("rknn_run failed: " + std::to_string(ret));
    }

    std::vector<rknn_output> fetched(impl_->outputs.size());
    for (std::size_t i = 0; i < fetched.size(); ++i) {
        std::memset(&fetched[i], 0, sizeof(fetched[i]));
        fetched[i].want_float = 1;
        fetched[i].is_prealloc = 1;
        fetched[i].index = static_cast<uint32_t>(i);
        fetched[i].buf = impl_->host[i].data();
        fetched[i].size = static_cast<uint32_t>(impl_->host[i].size() * sizeof(float));
    }
    ret = rknn_outputs_get(impl_->ctx, static_cast<uint32_t>(fetched.size()), fetched.data(), nullptr);
    if (ret != RKNN_SUCC) {
        return Status::Fail("rknn_outputs_get failed: " + std::to_string(ret));
    }

    std::vector<RawTensor> tensors(impl_->outputs.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const rknn_tensor_attr& attr = impl_->outputs[i];
        tensors[i].channel_last = attr.fmt == RKNN_TENSOR_NHWC;
        tensors[i].data = impl_->host[i].data();
        tensors[i].n_elems = attr.n_elems;
        tensors[i].dims.reserve(attr.n_dims);
        for (uint32_t d = 0; d < attr.n_dims; ++d) {
            tensors[i].dims.push_back(static_cast<int>(attr.dims[d]));
        }
    }

    DecodeParams params;
    params.num_classes = config_.num_classes;
    params.conf_threshold = config_.conf_threshold;
    params.nms_threshold = config_.nms_threshold;
    params.max_detections = config_.max_detections;
    params.anchor_scores_are_probabilities = config_.anchor_scores_are_probabilities;
    params.dfl_scores_are_logits = config_.dfl_scores_are_logits;
    params.letterbox = input.meta;
    const DecodeResult decoded = decode_yolov8(tensors, params);
    rknn_outputs_release(impl_->ctx, static_cast<uint32_t>(fetched.size()), fetched.data());

    const auto t1 = std::chrono::steady_clock::now();
    result.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!decoded.layout_recognized) {
        if (!impl_->layout_warned) {
            YR_LOGE("YOLO output layout is not the supported anchor head or 3-scale DFL head");
            impl_->layout_warned = true;
        }
        return Status::Fail("unrecognized yolo output layout");
    }
    result.detections = decoded.detections;
    return Status::Ok();
}

}  // namespace yolo_rknn
