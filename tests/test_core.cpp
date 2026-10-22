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

void test_yolov3_head() {
    constexpr int kClasses = 1;
    constexpr int kChannels = 18;
    constexpr int kGrid = 2;
    std::vector<float> data(static_cast<std::size_t>(kChannels * kGrid * kGrid), -8.f);
    auto at = [&](int c, int y, int x) -> float& {
        return data[static_cast<std::size_t>((c * kGrid + y) * kGrid + x)];
    };
    at(0, 1, 1) = 0.f;
    at(1, 1, 1) = 0.f;
    at(2, 1, 1) = 0.f;
    at(3, 1, 1) = 0.f;
    at(4, 1, 1) = 8.f;
    at(5, 1, 1) = 8.f;

    yolo_rknn::RawTensor tensor;
    tensor.dims.push_back(kGrid);
    tensor.dims.push_back(kGrid);
    tensor.dims.push_back(kChannels);
    tensor.dims.push_back(1);
    tensor.channel_last = false;
    tensor.data = data.data();
    tensor.n_elems = data.size();

    yolo_rknn::DecodeParams params;
    params.num_classes = kClasses;
    params.conf_threshold = 0.25f;
    params.nms_threshold = 0.45f;
    params.max_detections = 10;
    params.anchors = {10.f, 10.f, 20.f, 20.f, 30.f, 30.f};
    params.letterbox = identity_box(32);

    const yolo_rknn::DecodeResult result =
        yolo_rknn::decode_yolov3(std::vector<yolo_rknn::RawTensor>(1, tensor), params);
    expect(result.layout_recognized, "yolov3 layout");
    expect(result.detections.size() == 1, "one yolov3 box");
    if (!result.detections.empty()) {
        // sigmoid(0)=0.5，中心 (0.5+1)*16=24，anchor 10，左上角 19。
        expect_near(result.detections[0].x, 19.f, 0.2f, "yolov3 x");
        expect_near(result.detections[0].y, 19.f, 0.2f, "yolov3 y");
        expect_near(result.detections[0].w, 10.f, 0.2f, "yolov3 w");
        expect_near(result.detections[0].h, 10.f, 0.2f, "yolov3 h");
        expect(result.detections[0].score > 0.9f, "yolov3 score");
    }
}

void test_yolov5_split_dfl() {
    constexpr int kGrid = 2;
    std::vector<float> box(static_cast<std::size_t>(64 * kGrid * kGrid), -20.f);
    std::vector<float> early(static_cast<std::size_t>(kGrid * kGrid), 0.99f);
    std::vector<float> final_score(static_cast<std::size_t>(kGrid * kGrid), 0.f);
    for (int side = 0; side < 4; ++side) {
        box[static_cast<std::size_t>((side * 16 + 1) * kGrid * kGrid)] = 20.f;
    }
    final_score[0] = 0.95f;

    auto grid = [&](std::vector<float>* storage, int channels) {
        yolo_rknn::RawTensor tensor;
        tensor.dims.push_back(1);
        tensor.dims.push_back(channels);
        tensor.dims.push_back(kGrid);
        tensor.dims.push_back(kGrid);
        tensor.channel_last = false;
        tensor.data = storage->data();
        tensor.n_elems = storage->size();
        return tensor;
    };
    std::vector<yolo_rknn::RawTensor> tensors;
    tensors.push_back(grid(&box, 64));
    tensors.push_back(grid(&early, 1));
    tensors.push_back(grid(&final_score, 1));

    yolo_rknn::DecodeParams params;
    params.num_classes = 1;
    params.conf_threshold = 0.25f;
    params.nms_threshold = 0.45f;
    params.max_detections = 10;
    params.dfl_scores_are_logits = false;
    params.letterbox = identity_box(16);

    expect(yolo_rknn::yolov5_split_dfl_recognized(tensors, 1, 16, 16), "split dfl shape");
    const yolo_rknn::DecodeResult result = yolo_rknn::decode_yolov5(tensors, params);
    expect(result.layout_recognized, "yolov5 split layout");
    expect(result.detections.size() == 1, "one yolov5 box");
    if (!result.detections.empty()) {
        expect_near(result.detections[0].x, 0.f, 0.6f, "yolov5 x");
        expect_near(result.detections[0].y, 0.f, 0.6f, "yolov5 y");
        expect_near(result.detections[0].w, 12.f, 0.6f, "yolov5 w");
        expect_near(result.detections[0].h, 12.f, 0.6f, "yolov5 h");
        expect_near(result.detections[0].score, 0.95f, 1e-4f, "yolov5 uses the last score map");
    }
}

void test_input_fit() {
    yolo_rknn::ModelWindow window;
    expect(yolo_rknn::plan_model_window(yolo_rknn::InputFit::ShortSide, 1920, 1080, 640, 640, &window),
           "short side plan");
    expect(window.src_x == 420 && window.src_y == 0, "short side origin");
    expect(window.src_w == 1080 && window.src_h == 1080, "short side keeps the 1080 square");
    expect(window.dst_w == 640 && window.dst_h == 640 && window.dst_x == 0, "short side fills the model");
    expect(!window.fill, "short side has no gray bar");
    const float x0 = (0.f - window.meta.pad_x) / window.meta.scale;
    const float x1 = (640.f - window.meta.pad_x) / window.meta.scale;
    const float y1 = (640.f - window.meta.pad_y) / window.meta.scale;
    expect_near(x0, 420.f, 0.05f, "model left is source x=420");
    expect_near(x1, 1500.f, 0.05f, "model right is source x=1500");
    expect_near(y1, 1080.f, 0.05f, "model bottom is source y=1080");

    expect(yolo_rknn::plan_model_window(yolo_rknn::InputFit::ShortSide, 1080, 1920, 640, 640, &window),
           "portrait plan");
    expect(window.src_x == 0 && window.src_y == 420, "portrait crops the long side");
    expect(window.src_w == 1080 && window.src_h == 1080, "portrait window");

    expect(yolo_rknn::plan_model_window(yolo_rknn::InputFit::ShortSide, 1920, 1080, 640, 640, &window,
                                       yolo_rknn::WindowAnchor::Start),
           "left edge plan");
    expect(window.src_x == 0 && window.src_y == 0 && window.src_w == 1080 && window.src_h == 1080, "left window");
    expect_near((0.f - window.meta.pad_x) / window.meta.scale, 0.f, 0.05f, "left edge is source x=0");
    expect_near((640.f - window.meta.pad_x) / window.meta.scale, 1080.f, 0.05f, "left tile ends at x=1080");

    expect(yolo_rknn::plan_model_window(yolo_rknn::InputFit::ShortSide, 1920, 1080, 640, 640, &window,
                                       yolo_rknn::WindowAnchor::End),
           "right edge plan");
    expect(window.src_x == 840 && window.src_w == 1080 && window.src_h == 1080, "right window");
    expect_near((0.f - window.meta.pad_x) / window.meta.scale, 840.f, 0.05f, "right edge starts at x=840");
    expect_near((640.f - window.meta.pad_x) / window.meta.scale, 1920.f, 0.05f, "right edge is source x=1920");

    expect(yolo_rknn::plan_model_window(yolo_rknn::InputFit::Letterbox, 1920, 1080, 640, 640, &window),
           "letterbox plan");
    expect(window.src_x == 0 && window.src_w == 1920 && window.src_h == 1080, "letterbox uses the full frame");
    expect(window.fill, "letterbox fills gray");
    expect(window.dst_w == 640 && window.dst_y > 0, "letterbox pads the short side");
    expect_near(window.meta.pad_y, static_cast<float>(window.dst_y), 0.01f, "letterbox pad matches the gray bar");
}

void test_tile_duplicate() {
    yolo_rknn::Detection full;
    full.x = 1000.f;
    full.y = 200.f;
    full.w = 160.f;
    full.h = 200.f;
    full.score = 0.70f;
    full.class_id = 0;
    yolo_rknn::Detection clipped;
    clipped.x = 1010.f;
    clipped.y = 210.f;
    clipped.w = 70.f;
    clipped.h = 170.f;
    clipped.score = 0.92f;
    clipped.class_id = 0;
    yolo_rknn::Detection neighbor;
    neighbor.x = 1300.f;
    neighbor.y = 200.f;
    neighbor.w = 70.f;
    neighbor.h = 180.f;
    neighbor.score = 0.80f;
    neighbor.class_id = 0;

    const std::vector<yolo_rknn::Detection> kept =
        yolo_rknn::suppress_overlaps({clipped, full, neighbor}, 0.45f, 10);
    expect(kept.size() == 2, "clipped duplicate is merged, neighbor stays");
    if (kept.size() == 2) {
        const yolo_rknn::Detection* wide = kept[0].w > kept[1].w ? &kept[0] : &kept[1];
        const yolo_rknn::Detection* other = wide == &kept[0] ? &kept[1] : &kept[0];
        expect_near(wide->w, 160.f, 0.1f, "keep the unclipped box");
        expect_near(other->x, 1300.f, 0.1f, "side-by-side target stays");
    }
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
    test_yolov3_head();
    test_yolov5_split_dfl();
    test_input_fit();
    test_tile_duplicate();
    test_zoom();
    test_osd_stroke();
    if (g_failed != 0) {
        std::cerr << g_failed << " assertion(s) failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
