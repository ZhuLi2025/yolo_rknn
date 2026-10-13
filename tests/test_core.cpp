#include "yolo_rknn/business/zoom_policy.hpp"
#include "yolo_rknn/display/osd_painter.hpp"
#include "yolo_rknn/infer/postprocess.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

int g_failed = 0;

void expect(bool cond, const char* text) {
    if (!cond) {
        std::cerr << "FAIL " << text << "\n";
        ++g_failed;
    }
}

void expect_near(float got, float want, float tol, const char* text) {
    if (std::fabs(got - want) > tol) {
        std::cerr << "FAIL " << text << " got=" << got << " want=" << want << "\n";
        ++g_failed;
    }
}

yolo_rknn::LetterboxMeta identity_box(int size) {
    yolo_rknn::LetterboxMeta meta;
    meta.scale = 1.f;
    meta.pad_x = 0;
    meta.pad_y = 0;
    meta.src_width = size;
    meta.src_height = size;
    meta.model_width = size;
    meta.model_height = size;
    return meta;
}

void test_anchor_nms() {
    const int classes = 80;
    const int channels = 4 + classes;
    const int count = 2;
    std::vector<float> data(static_cast<std::size_t>(channels * count), 0.f);
    auto at = [&](int c, int n) -> float& { return data[static_cast<std::size_t>(c * count + n)]; };
    at(0, 0) = 320.f;
    at(1, 0) = 320.f;
    at(2, 0) = 100.f;
    at(3, 0) = 80.f;
    at(4, 0) = 0.9f;
    at(0, 1) = 330.f;
    at(1, 1) = 330.f;
    at(2, 1) = 100.f;
    at(3, 1) = 80.f;
    at(4, 1) = 0.4f;

    yolo_rknn::RawTensor tensor;
    tensor.dims.push_back(1);
    tensor.dims.push_back(channels);
    tensor.dims.push_back(count);
    tensor.channel_last = false;
    tensor.data = data.data();
    tensor.n_elems = data.size();

    yolo_rknn::DecodeParams params;
    params.num_classes = classes;
    params.conf_threshold = 0.25f;
    params.nms_threshold = 0.45f;
    params.max_detections = 10;
    params.anchor_scores_are_probabilities = true;
    params.letterbox = identity_box(640);

    const yolo_rknn::DecodeResult result = yolo_rknn::decode_yolov8(std::vector<yolo_rknn::RawTensor>(1, tensor), params);
    expect(result.layout_recognized, "anchor layout");
    expect(result.detections.size() == 1, "nms keeps one box");
    if (!result.detections.empty()) {
        expect(result.detections[0].class_id == 0, "class 0");
        expect_near(result.detections[0].score, 0.9f, 1e-4f, "score");
        expect_near(result.detections[0].x, 270.f, 0.2f, "x");
        expect_near(result.detections[0].y, 280.f, 0.2f, "y");
        expect_near(result.detections[0].w, 100.f, 0.2f, "w");
        expect_near(result.detections[0].h, 80.f, 0.2f, "h");
    }
}

void fill_dfl_cell(std::vector<float>* data, int channels, int height, int width, int x, int y, float class_logit) {
    for (int c = 0; c < channels; ++c) {
        (*data)[static_cast<std::size_t>((c * height + y) * width + x)] = -20.f;
    }
    for (int side = 0; side < 4; ++side) {
        (*data)[static_cast<std::size_t>(((side * 16 + 1) * height + y) * width + x)] = 20.f;
    }
    (*data)[static_cast<std::size_t>((64 * height + y) * width + x)] = class_logit;
}

yolo_rknn::RawTensor make_dfl(std::vector<float>* storage, int height, int width, float class_logit) {
    const int channels = 65;
    storage->assign(static_cast<std::size_t>(channels * height * width), -20.f);
    fill_dfl_cell(storage, channels, height, width, 0, 0, class_logit);
    yolo_rknn::RawTensor tensor;
    tensor.dims.push_back(1);
    tensor.dims.push_back(channels);
    tensor.dims.push_back(height);
    tensor.dims.push_back(width);
    tensor.channel_last = false;
    tensor.data = storage->data();
    tensor.n_elems = storage->size();
    return tensor;
}

void test_dfl() {
    std::vector<float> a, b, c;
    yolo_rknn::RawTensor t0 = make_dfl(&a, 2, 2, 8.f);
    yolo_rknn::RawTensor t1 = make_dfl(&b, 1, 1, -20.f);
    yolo_rknn::RawTensor t2 = make_dfl(&c, 1, 1, -20.f);
    std::vector<yolo_rknn::RawTensor> tensors;
    tensors.push_back(t0);
    tensors.push_back(t1);
    tensors.push_back(t2);

    yolo_rknn::DecodeParams params;
    params.num_classes = 1;
    params.conf_threshold = 0.25f;
    params.nms_threshold = 0.45f;
    params.max_detections = 10;
    params.dfl_scores_are_logits = true;
    params.letterbox = identity_box(16);

    const yolo_rknn::DecodeResult result = yolo_rknn::decode_yolov8(tensors, params);
    expect(result.layout_recognized, "dfl layout");
    expect(result.detections.size() == 1, "one dfl box");
    if (!result.detections.empty()) {
        // (0.5-1)*8 = -4，夹到图像内后是 0；右下 (0.5+1)*8 = 12。
        expect_near(result.detections[0].x, 0.f, 0.6f, "dfl x");
        expect_near(result.detections[0].y, 0.f, 0.6f, "dfl y");
        expect_near(result.detections[0].w, 12.f, 0.6f, "dfl w");
        expect_near(result.detections[0].h, 12.f, 0.6f, "dfl h");
        expect(result.detections[0].score > 0.9f, "dfl score");
    }
}

void test_unknown_layout() {
    yolo_rknn::RawTensor tensor;
    tensor.dims.push_back(1);
    tensor.dims.push_back(7);
    tensor.dims.push_back(3);
    float dummy[21] = {0};
    tensor.data = dummy;
    tensor.n_elems = 21;
    yolo_rknn::DecodeParams params;
    params.num_classes = 80;
    params.letterbox = identity_box(64);
    const yolo_rknn::DecodeResult result = yolo_rknn::decode_yolov8(std::vector<yolo_rknn::RawTensor>(1, tensor), params);
    expect(!result.layout_recognized, "unknown layout rejected");
}

void test_zoom() {
    yolo_rknn::ZoomInput in;
    in.has_target = false;
    expect(yolo_rknn::decide_zoom(in) == yolo_rknn::ZoomAction::Hold, "no target holds");

    in.has_target = true;
    in.target_ratio = 0.40f;
    in.deadband = 0.20f;
    in.bbox_h_ratio = 0.20f;
    expect(yolo_rknn::decide_zoom(in) == yolo_rknn::ZoomAction::ZoomIn, "small box zooms in");

    in.bbox_h_ratio = 0.40f;
    expect(yolo_rknn::decide_zoom(in) == yolo_rknn::ZoomAction::Hold, "on target holds");

    in.bbox_h_ratio = 0.70f;
    expect(yolo_rknn::decide_zoom(in) == yolo_rknn::ZoomAction::ZoomOut, "large box zooms out");

    in.bbox_h_ratio = 0.40f;
    in.height_valid = true;
    in.height_m = 20.f;
    in.reference_height_m = 10.f;
    expect(yolo_rknn::decide_zoom(in) == yolo_rknn::ZoomAction::Hold, "height does not wind the lens");

    in.bbox_h_ratio = 0.35f;
    expect(yolo_rknn::decide_zoom(in) == yolo_rknn::ZoomAction::ZoomIn, "height gain amplifies a small error");
}

void test_osd_stroke() {
    constexpr int kW = 64;
    constexpr int kH = 64;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kW * kH * 4), 0);
    yolo_rknn::OsdCanvas canvas;
    canvas.pixels = pixels.data();
    canvas.width = kW;
    canvas.height = kH;
    canvas.stride_bytes = kW * 4;

    yolo_rknn::Detection det;
    det.x = 8.f;
    det.y = 8.f;
    det.w = 40.f;
    det.h = 40.f;
    det.score = 0.9f;
    det.class_id = 0;
    yolo_rknn::osd_draw(canvas, {det});

    auto argb = [&](int x, int y) {
        std::uint32_t value = 0;
        std::memcpy(&value, pixels.data() + (static_cast<std::size_t>(y) * kW + x) * 4, 4);
        return value;
    };
    expect((argb(8, 8) >> 24) != 0, "box border is visible");
    expect(argb(28, 40) == 0, "box interior stays transparent");
}

}  // namespace

int main() {
    test_anchor_nms();
    test_dfl();
    test_unknown_layout();
    test_zoom();
    test_osd_stroke();
    if (g_failed != 0) {
        std::cerr << g_failed << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
