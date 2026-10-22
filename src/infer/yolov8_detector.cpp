#include "yolo_rknn/infer/yolov8_detector.hpp"

#include "yolo_rknn/common/logger.hpp"
#include "yolo_rknn/infer/postprocess.hpp"

#include <rknn_api.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
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

namespace {

rknn_core_mask core_mask_from_config(const std::string& name) {
    if (name == "0") {
        return RKNN_NPU_CORE_0;
    }
    if (name == "1") {
        return RKNN_NPU_CORE_1;
    }
    if (name == "2") {
        return RKNN_NPU_CORE_2;
    }
    if (name == "0_1") {
        return RKNN_NPU_CORE_0_1;
    }
    if (name == "0_1_2") {
        return RKNN_NPU_CORE_0_1_2;
    }
    return RKNN_NPU_CORE_AUTO;
}

}  // namespace

struct SideContext {
    rknn_context ctx = 0;
    rknn_tensor_attr input;
    rknn_tensor_mem* input_mem = nullptr;
    std::vector<rknn_tensor_attr> outputs;
    std::vector<std::vector<float>> host;
    bool pass_through = false;
    bool zero_copy = false;

    SideContext() { std::memset(&input, 0, sizeof(input)); }
};

struct YoloV8Detector::Impl {
    rknn_context ctx = 0;
    rknn_tensor_attr input;
    rknn_tensor_mem* input_mem = nullptr;
    std::vector<rknn_tensor_attr> outputs;
    std::vector<std::vector<float>> host;
    bool pass_through = false;
    bool zero_copy = false;
    bool layout_warned = false;
    bool parallel = false;
    bool parallel_logged = false;
    std::vector<float> anchors;
    SideContext side;

    Impl() { std::memset(&input, 0, sizeof(input)); }
};

Status query_side_io(rknn_context ctx, int width, int height, SideContext* side) {
    rknn_input_output_num io;
    std::memset(&io, 0, sizeof(io));
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != RKNN_SUCC || io.n_input < 1 || io.n_output < 1) {
        return Status::Fail("second context input/output count failed");
    }
    side->input.index = 0;
    if (rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &side->input, sizeof(side->input)) != RKNN_SUCC) {
        return Status::Fail("second context input attr failed");
    }
    if (!input_matches(side->input, width, height)) {
        return Status::Fail("second context input shape does not match the model");
    }
    const uint32_t rgb_bytes = static_cast<uint32_t>(width) * static_cast<uint32_t>(height) * 3u;
    side->pass_through = side->input.type == RKNN_TENSOR_UINT8 && side->input.fmt == RKNN_TENSOR_NHWC &&
                         side->input.size == rgb_bytes;
    side->outputs.resize(io.n_output);
    side->host.resize(io.n_output);
    for (uint32_t i = 0; i < io.n_output; ++i) {
        rknn_tensor_attr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.index = i;
        if (rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) != RKNN_SUCC) {
            return Status::Fail("second context output attr failed");
        }
        side->outputs[i] = attr;
        side->host[i].assign(attr.n_elems, 0.f);
    }
    return Status::Ok();
}

Status execute_context(const AppConfig& config, const std::vector<float>& anchors, rknn_context ctx, bool pass_through,
                       bool zero_copy, const std::vector<rknn_tensor_attr>& outputs,
                       std::vector<std::vector<float>>& host, const PreprocessResult& input,
                       std::vector<Detection>* dets) {
    if (ctx == 0) {
        return Status::Fail("detector is not initialized");
    }
    if (input.bytes == 0 || (input.rgb == nullptr && !zero_copy)) {
        return Status::Fail("empty preprocess buffer");
    }
    int ret = RKNN_SUCC;
    if (!zero_copy) {
        rknn_input feed;
        std::memset(&feed, 0, sizeof(feed));
        feed.index = 0;
        feed.buf = const_cast<std::uint8_t*>(input.rgb);
        feed.size = static_cast<uint32_t>(input.bytes);
        if (pass_through) {
            feed.pass_through = 1;
        } else {
            feed.pass_through = 0;
            feed.type = RKNN_TENSOR_UINT8;
            feed.fmt = RKNN_TENSOR_NHWC;
        }
        ret = rknn_inputs_set(ctx, 1, &feed);
        if (ret != RKNN_SUCC) {
            return Status::Fail("rknn_inputs_set failed: " + std::to_string(ret));
        }
    }
    ret = rknn_run(ctx, nullptr);
    if (ret != RKNN_SUCC) {
        return Status::Fail("rknn_run failed: " + std::to_string(ret));
    }

    std::vector<rknn_output> fetched(outputs.size());
    for (std::size_t i = 0; i < fetched.size(); ++i) {
        std::memset(&fetched[i], 0, sizeof(fetched[i]));
        fetched[i].want_float = 1;
        fetched[i].is_prealloc = 1;
        fetched[i].index = static_cast<uint32_t>(i);
        fetched[i].buf = host[i].data();
        fetched[i].size = static_cast<uint32_t>(host[i].size() * sizeof(float));
    }
    ret = rknn_outputs_get(ctx, static_cast<uint32_t>(fetched.size()), fetched.data(), nullptr);
    if (ret != RKNN_SUCC) {
        return Status::Fail("rknn_outputs_get failed: " + std::to_string(ret));
    }

    std::vector<RawTensor> tensors(outputs.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const rknn_tensor_attr& attr = outputs[i];
        tensors[i].channel_last = attr.fmt == RKNN_TENSOR_NHWC;
        tensors[i].data = host[i].data();
        tensors[i].n_elems = attr.n_elems;
        tensors[i].dims.reserve(attr.n_dims);
        for (uint32_t d = 0; d < attr.n_dims; ++d) {
            tensors[i].dims.push_back(static_cast<int>(attr.dims[d]));
        }
    }

    DecodeParams params;
    params.num_classes = config.num_classes;
    params.conf_threshold = config.conf_threshold;
    params.nms_threshold = config.nms_threshold;
    params.max_detections = config.max_detections;
    params.anchor_scores_are_probabilities = config.anchor_scores_are_probabilities;
    params.dfl_scores_are_logits = config.dfl_scores_are_logits;
    params.anchors = anchors;
    params.letterbox = input.meta;
    DecodeResult decoded;
    if (config.model_type == "yolov3") {
        decoded = decode_yolov3(tensors, params);
    } else if (config.model_type == "yolov5") {
        decoded = decode_yolov5(tensors, params);
    } else {
        decoded = decode_yolov8(tensors, params);
    }
    rknn_outputs_release(ctx, static_cast<uint32_t>(fetched.size()), fetched.data());
    if (!decoded.layout_recognized) {
        return Status::Fail("unrecognized yolo output layout");
    }
    *dets = std::move(decoded.detections);
    return Status::Ok();
}

YoloV8Detector::YoloV8Detector(const AppConfig& config) : config_(config), impl_(new Impl) {}

YoloV8Detector::~YoloV8Detector() {
    if (impl_ == nullptr) {
        return;
    }
    if (impl_->side.ctx) {
        if (impl_->side.input_mem) {
            rknn_destroy_mem(impl_->side.ctx, impl_->side.input_mem);
            impl_->side.input_mem = nullptr;
        }
        rknn_destroy(impl_->side.ctx);
        impl_->side.ctx = 0;
    }
    if (impl_->ctx) {
        if (impl_->input_mem) {
            rknn_destroy_mem(impl_->ctx, impl_->input_mem);
            impl_->input_mem = nullptr;
        }
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
    // RKNN2 的 rknn_init 多一个 extend 参数，传空表示模型就在这块内存里。
    const int ret = rknn_init(&impl_->ctx, model.data(), static_cast<uint32_t>(model.size()), RKNN_FLAG_PRIOR_HIGH,
                              nullptr);
    if (ret != RKNN_SUCC) {
        return Status::Fail("rknn_init failed: " + std::to_string(ret) +
                            " (model must be converted for rk3588 with RKNN-Toolkit2)");
    }

    const bool parallel = config_.batch_size == 2 && config_.preprocess == "short_side";
    if (config_.batch_size == 2 && !parallel) {
        YR_LOGW("batch_size=2 is ignored unless preprocess=short_side; using one NPU context");
    }
    if (parallel) {
        const int core_ret = rknn_set_core_mask(impl_->ctx, RKNN_NPU_CORE_0);
        if (core_ret != RKNN_SUCC) {
            return Status::Fail("rknn_set_core_mask core0 failed: " + std::to_string(core_ret));
        }
        YR_LOGI("parallel batch=2: context0 on NPU core0");
    } else {
        const rknn_core_mask cores = core_mask_from_config(config_.npu_cores);
        const int core_ret = rknn_set_core_mask(impl_->ctx, cores);
        if (core_ret != RKNN_SUCC) {
            YR_LOGW("rknn_set_core_mask %s -> %d", config_.npu_cores.c_str(), core_ret);
        } else {
            YR_LOGI("rknn cores=%s", config_.npu_cores.c_str());
        }
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
        return Status::Fail("model input shape does not match model_width x model_height (" +
                            std::to_string(config_.model_width) + "x" + std::to_string(config_.model_height) +
                            "): " + shape_string(impl_->input));
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

    if (config_.model_type == "yolov3") {
        std::vector<RawTensor> shapes(impl_->outputs.size());
        for (std::size_t i = 0; i < impl_->outputs.size(); ++i) {
            const rknn_tensor_attr& attr = impl_->outputs[i];
            shapes[i].channel_last = attr.fmt == RKNN_TENSOR_NHWC;
            shapes[i].n_elems = attr.n_elems;
            for (uint32_t d = 0; d < attr.n_dims; ++d) {
                shapes[i].dims.push_back(static_cast<int>(attr.dims[d]));
            }
        }
        if (!yolov3_outputs_recognized(shapes, config_.num_classes, config_.model_width, config_.model_height)) {
            return Status::Fail("model_type=yolov3 but outputs are not darknet YOLOv3 heads (3 anchors, 5+classes)");
        }
        if (config_.anchors.empty()) {
            // COCO 默认值，顺序是 stride 从大到小，每个 stride 三个 anchor。
            if (impl_->outputs.size() == 2) {
                const float tiny[] = {81.f, 82.f, 135.f, 169.f, 344.f, 319.f, 10.f, 14.f, 23.f, 27.f, 37.f, 58.f};
                impl_->anchors.assign(tiny, tiny + 12);
            } else if (impl_->outputs.size() == 3) {
                const float full[] = {116.f, 90.f,  156.f, 198.f, 373.f, 326.f, 30.f, 61.f, 62.f, 45.f,
                                      59.f,  119.f, 10.f,   13.f,  16.f,  30.f,  33.f, 23.f};
                impl_->anchors.assign(full, full + 18);
            } else {
                return Status::Fail("yolov3 anchors are empty and this output count has no built-in default");
            }
            YR_LOGI("yolov3 using built-in COCO anchors for %zu heads", impl_->outputs.size());
        } else {
            impl_->anchors = config_.anchors;
        }
        if (impl_->anchors.size() != impl_->outputs.size() * 6) {
            return Status::Fail("anchors has " + std::to_string(impl_->anchors.size() / 2) +
                                " pairs, model has " + std::to_string(impl_->outputs.size()) +
                                " heads, need 3 anchors per head");
        }
    } else if (config_.model_type == "yolov5") {
        std::vector<RawTensor> shapes(impl_->outputs.size());
        for (std::size_t i = 0; i < impl_->outputs.size(); ++i) {
            const rknn_tensor_attr& attr = impl_->outputs[i];
            shapes[i].channel_last = attr.fmt == RKNN_TENSOR_NHWC;
            shapes[i].n_elems = attr.n_elems;
            for (uint32_t d = 0; d < attr.n_dims; ++d) {
                shapes[i].dims.push_back(static_cast<int>(attr.dims[d]));
            }
        }
        if (yolov5_split_dfl_recognized(shapes, config_.num_classes, config_.model_width, config_.model_height)) {
            YR_LOGI("yolov5 split DFL heads, classes=%d", config_.num_classes);
        } else if (yolov3_outputs_recognized(shapes, config_.num_classes, config_.model_width,
                                             config_.model_height)) {
            if (config_.anchors.empty()) {
                // COCO YOLOv5，stride 从大到小：32、16、8，每个 stride 三个 anchor。
                const float coco[] = {116.f, 90.f, 156.f, 198.f, 373.f, 326.f, 30.f, 61.f, 62.f,
                                      45.f,  59.f, 119.f, 10.f,  13.f,  16.f,  30.f, 33.f, 23.f};
                impl_->anchors.assign(coco, coco + 18);
                YR_LOGI("yolov5 using built-in COCO anchors");
            } else {
                impl_->anchors = config_.anchors;
            }
            if (impl_->anchors.size() != impl_->outputs.size() * 6) {
                return Status::Fail("anchors has " + std::to_string(impl_->anchors.size() / 2) +
                                    " pairs, model has " + std::to_string(impl_->outputs.size()) +
                                    " heads, need 3 anchors per head");
            }
        } else {
            return Status::Fail(
                "model_type=yolov5 but outputs are neither split DFL maps nor 3-anchor heads");
        }
    }
    YR_LOGI("model_type=%s", config_.model_type.c_str());
    if (!parallel) {
        return Status::Ok();
    }

    int dup = rknn_dup_context(&impl_->ctx, &impl_->side.ctx);
    if (dup != RKNN_SUCC) {
        YR_LOGW("rknn_dup_context -> %d, loading the model into a second context", dup);
        impl_->side.ctx = 0;
        dup = rknn_init(&impl_->side.ctx, model.data(), static_cast<uint32_t>(model.size()), RKNN_FLAG_PRIOR_HIGH,
                        nullptr);
    }
    if (dup != RKNN_SUCC || impl_->side.ctx == 0) {
        return Status::Fail("second rknn context failed: " + std::to_string(dup));
    }
    const int side_core = rknn_set_core_mask(impl_->side.ctx, RKNN_NPU_CORE_1);
    if (side_core != RKNN_SUCC) {
        return Status::Fail("rknn_set_core_mask core1 failed: " + std::to_string(side_core));
    }
    Status side_io = query_side_io(impl_->side.ctx, config_.model_width, config_.model_height, &impl_->side);
    if (!side_io.ok()) {
        return side_io;
    }
    if (impl_->side.outputs.size() != impl_->outputs.size()) {
        return Status::Fail("second context output count does not match");
    }
    for (std::size_t i = 0; i < impl_->outputs.size(); ++i) {
        if (impl_->side.outputs[i].n_elems != impl_->outputs[i].n_elems) {
            return Status::Fail("second context output shape does not match");
        }
    }
    impl_->parallel = true;
    YR_LOGI("parallel batch=2 ready on NPU core0+core1, model file stays batch 1");
    return Status::Ok();
}

void bind_one(rknn_context ctx, rknn_tensor_attr input_attr, bool pass_through, int width, int height, int fd,
              void* virt, std::size_t bytes, rknn_tensor_mem** mem_out, bool* zero_copy, int tile) {
    if (ctx == 0 || *zero_copy || fd < 0 || virt == nullptr) {
        return;
    }
    const std::size_t packed = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u;
    if (bytes < packed) {
        YR_LOGW("preprocess buffer %zu is smaller than RGB888 %zu, keep rknn_inputs_set", bytes, packed);
        return;
    }
    rknn_tensor_mem* mem = rknn_create_mem_from_fd(ctx, fd, virt, static_cast<uint32_t>(bytes), 0);
    if (mem == nullptr) {
        YR_LOGW("rknn_create_mem_from_fd failed, keep rknn_inputs_set");
        return;
    }
    rknn_tensor_attr attr = input_attr;
    const bool tight_stride = (attr.w_stride == 0 || attr.w_stride == static_cast<uint32_t>(width)) &&
                              (attr.h_stride == 0 || attr.h_stride == static_cast<uint32_t>(height));
    if (pass_through && tight_stride) {
        attr.pass_through = 1;
    } else {
        attr.pass_through = 0;
        attr.type = RKNN_TENSOR_UINT8;
        attr.fmt = RKNN_TENSOR_NHWC;
    }
    const int ret = rknn_set_io_mem(ctx, mem, &attr);
    if (ret != RKNN_SUCC) {
        YR_LOGW("rknn_set_io_mem failed: %d, keep rknn_inputs_set", ret);
        rknn_destroy_mem(ctx, mem);
        return;
    }
    *mem_out = mem;
    *zero_copy = true;
    YR_LOGI("rknn input zero-copy tile=%d fd=%d pass_through=%u bytes=%zu", tile, fd, attr.pass_through, bytes);
}

void YoloV8Detector::bind_input(int fd, void* virt, std::size_t bytes) { bind_input(0, fd, virt, bytes); }

void YoloV8Detector::bind_input(int tile, int fd, void* virt, std::size_t bytes) {
    if (tile == 0) {
        bind_one(impl_->ctx, impl_->input, impl_->pass_through, config_.model_width, config_.model_height, fd, virt,
                 bytes, &impl_->input_mem, &impl_->zero_copy, 0);
        return;
    }
    if (tile == 1 && impl_->parallel) {
        bind_one(impl_->side.ctx, impl_->side.input, impl_->side.pass_through, config_.model_width,
                 config_.model_height, fd, virt, bytes, &impl_->side.input_mem, &impl_->side.zero_copy, 1);
    }
}

Status YoloV8Detector::infer(const PreprocessResult& input, InferenceResult& result) {
    return infer(&input, 1, result);
}

Status YoloV8Detector::infer(const PreprocessResult* tiles, int tile_count, InferenceResult& result) {
    if (tiles == nullptr || tile_count < 1) {
        return Status::Fail("empty preprocess batch");
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Detection> left;
    std::vector<Detection> right;
    Status left_st = Status::Ok();
    Status right_st = Status::Ok();
    if (tile_count >= 2 && impl_->parallel) {
        std::thread core0([&] {
            left_st = execute_context(config_, impl_->anchors, impl_->ctx, impl_->pass_through, impl_->zero_copy,
                                      impl_->outputs, impl_->host, tiles[0], &left);
        });
        std::thread core1([&] {
            right_st = execute_context(config_, impl_->anchors, impl_->side.ctx, impl_->side.pass_through,
                                       impl_->side.zero_copy, impl_->side.outputs, impl_->side.host, tiles[1],
                                       &right);
        });
        core0.join();
        core1.join();
    } else {
        left_st = execute_context(config_, impl_->anchors, impl_->ctx, impl_->pass_through, impl_->zero_copy,
                                  impl_->outputs, impl_->host, tiles[0], &left);
    }
    const auto t1 = std::chrono::steady_clock::now();
    result.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!left_st.ok()) {
        if (left_st.message() == "unrecognized yolo output layout" && !impl_->layout_warned) {
            YR_LOGE("YOLO output layout does not match model_type=%s", config_.model_type.c_str());
            impl_->layout_warned = true;
        }
        return left_st;
    }
    if (!right_st.ok()) {
        if (right_st.message() == "unrecognized yolo output layout" && !impl_->layout_warned) {
            YR_LOGE("YOLO output layout does not match model_type=%s", config_.model_type.c_str());
            impl_->layout_warned = true;
        }
        return right_st;
    }
    if (tile_count >= 2 && impl_->parallel) {
        left.insert(left.end(), right.begin(), right.end());
        result.detections =
            suppress_overlaps(std::move(left), config_.nms_threshold, config_.max_detections);
        if (!impl_->parallel_logged) {
            YR_LOGI("parallel batch=2 infer %.1f ms dets=%zu", result.infer_ms, result.detections.size());
            impl_->parallel_logged = true;
        }
    } else {
        result.detections = std::move(left);
    }
    return Status::Ok();
}

}  // namespace yolo_rknn
