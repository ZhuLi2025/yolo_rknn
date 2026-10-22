#include "yolo_rknn/infer/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace yolo_rknn {
namespace {

float clampf(float value, float lo, float hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

float sigmoid(float x) {
    if (x >= 0.f) {
        const float z = std::exp(-x);
        return 1.f / (1.f + z);
    }
    const float z = std::exp(x);
    return z / (1.f + z);
}

float dfl_expect(const float* logits, int n) {
    float max_logit = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    float sum = 0.f;
    float acc = 0.f;
    for (int i = 0; i < n; ++i) {
        const float e = std::exp(logits[i] - max_logit);
        sum += e;
        acc += e * static_cast<float>(i);
    }
    if (sum <= 0.f) {
        return 0.f;
    }
    return acc / sum;
}

float iou(const Detection& a, const Detection& b) {
    const float ax2 = a.x + a.w;
    const float ay2 = a.y + a.h;
    const float bx2 = b.x + b.w;
    const float by2 = b.y + b.h;
    const float ix1 = std::max(a.x, b.x);
    const float iy1 = std::max(a.y, b.y);
    const float ix2 = std::min(ax2, bx2);
    const float iy2 = std::min(ay2, by2);
    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    if (iw <= 0.f || ih <= 0.f) {
        return 0.f;
    }
    const float inter = iw * ih;
    const float uni = a.w * a.h + b.w * b.h - inter;
    if (uni <= 0.f) {
        return 0.f;
    }
    return inter / uni;
}

float intersection_over_smaller(const Detection& a, const Detection& b) {
    const float ax2 = a.x + a.w;
    const float ay2 = a.y + a.h;
    const float bx2 = b.x + b.w;
    const float by2 = b.y + b.h;
    const float iw = std::min(ax2, bx2) - std::max(a.x, b.x);
    const float ih = std::min(ay2, by2) - std::max(a.y, b.y);
    if (iw <= 0.f || ih <= 0.f) {
        return 0.f;
    }
    const float smaller = std::min(a.w * a.h, b.w * b.h);
    if (smaller <= 0.f) {
        return 0.f;
    }
    return (iw * ih) / smaller;
}

bool center_inside(const Detection& inner, const Detection& outer) {
    const float cx = inner.x + inner.w * 0.5f;
    const float cy = inner.y + inner.h * 0.5f;
    return cx >= outer.x && cx <= outer.x + outer.w && cy >= outer.y && cy <= outer.y + outer.h;
}

// 切片交界上，一边常是被裁断的小框，和另一边的完整框 IoU 会掉到阈值以下。
// 小框大部分落在大框里，或任一框的中心落在另一框内，都当成同一个目标。
bool same_tiled_target(const Detection& a, const Detection& b, float iou_threshold) {
    const float overlap = iou(a, b);
    const float covered = intersection_over_smaller(a, b);
    if (overlap > iou_threshold || covered >= 0.55f) {
        return true;
    }
    return covered >= 0.30f && (center_inside(a, b) || center_inside(b, a));
}

void map_box(float x1, float y1, float x2, float y2, const LetterboxMeta& meta, Detection* det) {
    if (meta.scale <= 0.f) {
        return;
    }
    x1 = (x1 - meta.pad_x) / meta.scale;
    y1 = (y1 - meta.pad_y) / meta.scale;
    x2 = (x2 - meta.pad_x) / meta.scale;
    y2 = (y2 - meta.pad_y) / meta.scale;
    const float max_w = static_cast<float>(std::max(meta.src_width, 1));
    const float max_h = static_cast<float>(std::max(meta.src_height, 1));
    x1 = clampf(x1, 0.f, max_w);
    y1 = clampf(y1, 0.f, max_h);
    x2 = clampf(x2, 0.f, max_w);
    y2 = clampf(y2, 0.f, max_h);
    det->x = x1;
    det->y = y1;
    det->w = x2 - x1;
    det->h = y2 - y1;
}

std::vector<Detection> nms(std::vector<Detection> dets, float threshold, int max_det) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    std::vector<char> removed(dets.size(), 0);
    std::vector<Detection> kept;
    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) {
            continue;
        }
        kept.push_back(dets[i]);
        if (static_cast<int>(kept.size()) >= max_det) {
            break;
        }
        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j] || dets[j].class_id != dets[i].class_id) {
                continue;
            }
            if (iou(dets[i], dets[j]) > threshold) {
                removed[j] = 1;
            }
        }
    }
    return kept;
}

bool anchor_shape(const RawTensor& tensor, int num_classes, int* channels, int* count, bool* channel_first) {
    const int expect = 4 + num_classes;
    if (expect <= 4) {
        return false;
    }
    if (tensor.dims.size() == 3 && tensor.dims[0] == 1) {
        if (tensor.dims[1] == expect) {
            *channels = tensor.dims[1];
            *count = tensor.dims[2];
            *channel_first = true;
            return true;
        }
        if (tensor.dims[2] == expect) {
            *channels = tensor.dims[2];
            *count = tensor.dims[1];
            *channel_first = false;
            return true;
        }
    }
    if (tensor.dims.size() == 2) {
        if (tensor.dims[0] == expect) {
            *channels = tensor.dims[0];
            *count = tensor.dims[1];
            *channel_first = true;
            return true;
        }
        if (tensor.dims[1] == expect) {
            *channels = tensor.dims[1];
            *count = tensor.dims[0];
            *channel_first = false;
            return true;
        }
    }
    return false;
}

float anchor_at(const float* data, int channels, int count, bool channel_first, int c, int i) {
    if (channel_first) {
        return data[c * count + i];
    }
    return data[i * channels + c];
}

void decode_anchor(const RawTensor& tensor, const DecodeParams& params, std::vector<Detection>* dets) {
    int channels = 0;
    int count = 0;
    bool channel_first = false;
    if (!anchor_shape(tensor, params.num_classes, &channels, &count, &channel_first)) {
        return;
    }
    if (tensor.data == nullptr || count <= 0) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        float best = -1.f;
        int best_id = -1;
        for (int c = 0; c < params.num_classes; ++c) {
            float score = anchor_at(tensor.data, channels, count, channel_first, 4 + c, i);
            if (!params.anchor_scores_are_probabilities) {
                score = sigmoid(score);
            }
            if (score > best) {
                best = score;
                best_id = c;
            }
        }
        if (best < params.conf_threshold || best_id < 0) {
            continue;
        }
        const float cx = anchor_at(tensor.data, channels, count, channel_first, 0, i);
        const float cy = anchor_at(tensor.data, channels, count, channel_first, 1, i);
        const float bw = anchor_at(tensor.data, channels, count, channel_first, 2, i);
        const float bh = anchor_at(tensor.data, channels, count, channel_first, 3, i);
        Detection det;
        det.score = best;
        det.class_id = best_id;
        map_box(cx - bw * 0.5f, cy - bh * 0.5f, cx + bw * 0.5f, cy + bh * 0.5f, params.letterbox, &det);
        if (det.w >= 1.f && det.h >= 1.f) {
            dets->push_back(det);
        }
    }
}

float dfl_at(const RawTensor& tensor, int c, int y, int x, int channels, int height, int width) {
    if (tensor.channel_last) {
        return tensor.data[(y * width + x) * channels + c];
    }
    return tensor.data[(c * height + y) * width + x];
}

bool decode_dfl_head(const RawTensor& tensor, const DecodeParams& params, std::vector<Detection>* dets) {
    if (tensor.dims.size() != 4 || tensor.dims[0] != 1 || tensor.data == nullptr) {
        return false;
    }
    int channels = 0;
    int height = 0;
    int width = 0;
    if (tensor.channel_last) {
        height = tensor.dims[1];
        width = tensor.dims[2];
        channels = tensor.dims[3];
    } else {
        channels = tensor.dims[1];
        height = tensor.dims[2];
        width = tensor.dims[3];
    }
    if (height <= 0 || width <= 0 || channels < 65 || params.letterbox.model_width <= 0) {
        return false;
    }
    if (params.letterbox.model_width % width != 0) {
        return false;
    }
    const int stride = params.letterbox.model_width / width;
    int classes = channels - 64;
    if (params.num_classes > 0 && params.num_classes < classes) {
        classes = params.num_classes;
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float best = -1.f;
            int best_id = -1;
            for (int c = 0; c < classes; ++c) {
                float score = dfl_at(tensor, 64 + c, y, x, channels, height, width);
                if (params.dfl_scores_are_logits) {
                    score = sigmoid(score);
                }
                if (score > best) {
                    best = score;
                    best_id = c;
                }
            }
            if (best < params.conf_threshold || best_id < 0) {
                continue;
            }
            float dist[4];
            for (int side = 0; side < 4; ++side) {
                float bins[16];
                for (int b = 0; b < 16; ++b) {
                    bins[b] = dfl_at(tensor, side * 16 + b, y, x, channels, height, width);
                }
                dist[side] = dfl_expect(bins, 16);
            }
            const float cx = static_cast<float>(x) + 0.5f;
            const float cy = static_cast<float>(y) + 0.5f;
            const float s = static_cast<float>(stride);
            Detection det;
            det.score = best;
            det.class_id = best_id;
            map_box((cx - dist[0]) * s, (cy - dist[1]) * s, (cx + dist[2]) * s, (cy + dist[3]) * s,
                    params.letterbox, &det);
            if (det.w >= 1.f && det.h >= 1.f) {
                dets->push_back(det);
            }
        }
    }
    return true;
}

struct YoloV3Head {
    int grid_w = 0;
    int grid_h = 0;
    int channels = 0;
    int anchor_count = 0;
    int stride = 0;
    bool nchw = true;
    int order = 0;
};

bool view_yolov3_head(const RawTensor& tensor, int num_classes, int model_w, int model_h, YoloV3Head* head) {
    const int per = 5 + num_classes;
    if (per <= 5 || tensor.dims.size() != 4 || model_w <= 0 || model_h <= 0) {
        return false;
    }
    auto channels_ok = [&](int channels) {
        return channels > 0 && channels % per == 0 && channels / per >= 1 && channels / per <= 5;
    };
    int grid_w = 0;
    int grid_h = 0;
    int channels = 0;
    bool nchw = true;
    if (!tensor.channel_last && tensor.dims[0] == 1 && channels_ok(tensor.dims[1])) {
        channels = tensor.dims[1];
        grid_h = tensor.dims[2];
        grid_w = tensor.dims[3];
        nchw = true;
    } else if (tensor.channel_last && tensor.dims[0] == 1 && channels_ok(tensor.dims[3])) {
        grid_h = tensor.dims[1];
        grid_w = tensor.dims[2];
        channels = tensor.dims[3];
        nchw = false;
    } else if (!tensor.channel_last && tensor.dims[3] == 1 && channels_ok(tensor.dims[2])) {
        // Toolkit 1.x 查询到的 dims 是倒序：[W, H, C, N]，内存仍是 NCHW 平面。
        grid_w = tensor.dims[0];
        grid_h = tensor.dims[1];
        channels = tensor.dims[2];
        nchw = true;
    } else {
        return false;
    }
    if (grid_w < 1 || grid_h < 1 || model_w % grid_w != 0 || model_h % grid_h != 0) {
        return false;
    }
    if (model_w / grid_w != model_h / grid_h) {
        return false;
    }
    const int cells = grid_w * grid_h * channels;
    if (tensor.n_elems != 0 && static_cast<int>(tensor.n_elems) < cells) {
        return false;
    }
    head->grid_w = grid_w;
    head->grid_h = grid_h;
    head->channels = channels;
    head->anchor_count = channels / per;
    head->stride = model_w / grid_w;
    head->nchw = nchw;
    return true;
}

float yolov3_at(const float* data, const YoloV3Head& head, int channel, int y, int x) {
    if (head.nchw) {
        return data[(channel * head.grid_h + y) * head.grid_w + x];
    }
    return data[(y * head.grid_w + x) * head.channels + channel];
}

void decode_anchor_head(const RawTensor& tensor, const YoloV3Head& head, const float* anchor_wh,
                        const DecodeParams& params, bool yolov5, std::vector<Detection>* dets) {
    if (tensor.data == nullptr) {
        return;
    }
    const int per = 5 + params.num_classes;
    const float stride = static_cast<float>(head.stride);
    for (int a = 0; a < head.anchor_count; ++a) {
        const float anchor_w = anchor_wh[a * 2];
        const float anchor_h = anchor_wh[a * 2 + 1];
        const int base = a * per;
        for (int y = 0; y < head.grid_h; ++y) {
            for (int x = 0; x < head.grid_w; ++x) {
                const float objectness = sigmoid(yolov3_at(tensor.data, head, base + 4, y, x));
                if (objectness < params.conf_threshold) {
                    continue;
                }
                float best = 0.f;
                int best_id = -1;
                for (int c = 0; c < params.num_classes; ++c) {
                    const float score = objectness * sigmoid(yolov3_at(tensor.data, head, base + 5 + c, y, x));
                    if (score > best) {
                        best = score;
                        best_id = c;
                    }
                }
                if (best < params.conf_threshold || best_id < 0) {
                    continue;
                }
                float cx = 0.f;
                float cy = 0.f;
                float bw = 0.f;
                float bh = 0.f;
                if (yolov5) {
                    const float tx = sigmoid(yolov3_at(tensor.data, head, base + 0, y, x));
                    const float ty = sigmoid(yolov3_at(tensor.data, head, base + 1, y, x));
                    const float tw = sigmoid(yolov3_at(tensor.data, head, base + 2, y, x));
                    const float th = sigmoid(yolov3_at(tensor.data, head, base + 3, y, x));
                    cx = (tx * 2.f - 0.5f + static_cast<float>(x)) * stride;
                    cy = (ty * 2.f - 0.5f + static_cast<float>(y)) * stride;
                    bw = (tw * 2.f) * (tw * 2.f) * anchor_w;
                    bh = (th * 2.f) * (th * 2.f) * anchor_h;
                } else {
                    const float tx = sigmoid(yolov3_at(tensor.data, head, base + 0, y, x));
                    const float ty = sigmoid(yolov3_at(tensor.data, head, base + 1, y, x));
                    const float tw = clampf(yolov3_at(tensor.data, head, base + 2, y, x), -16.f, 16.f);
                    const float th = clampf(yolov3_at(tensor.data, head, base + 3, y, x), -16.f, 16.f);
                    cx = (tx + static_cast<float>(x)) * stride;
                    cy = (ty + static_cast<float>(y)) * stride;
                    bw = std::exp(tw) * anchor_w;
                    bh = std::exp(th) * anchor_h;
                }
                Detection det;
                det.score = best;
                det.class_id = best_id;
                map_box(cx - bw * 0.5f, cy - bh * 0.5f, cx + bw * 0.5f, cy + bh * 0.5f, params.letterbox, &det);
                if (det.w >= 1.f && det.h >= 1.f) {
                    dets->push_back(det);
                }
            }
        }
    }
}

struct SplitScale {
    int box = -1;
    int score = -1;
    int height = 0;
    int width = 0;
};

bool grid_hwc(const RawTensor& tensor, int* channels, int* height, int* width) {
    if (tensor.dims.size() != 4 || tensor.dims[0] != 1) {
        return false;
    }
    if (tensor.channel_last) {
        *height = tensor.dims[1];
        *width = tensor.dims[2];
        *channels = tensor.dims[3];
    } else {
        *channels = tensor.dims[1];
        *height = tensor.dims[2];
        *width = tensor.dims[3];
    }
    return *channels > 0 && *height > 0 && *width > 0;
}

// 同一尺度上，64 通道是 DFL 框，nc 通道是分数。导出时 Sigmoid 和 Clip 都会露出来，取最后一张分数。
bool match_split_dfl(const std::vector<RawTensor>& tensors, int num_classes, int model_w, int model_h,
                     std::vector<SplitScale>* scales) {
    if (tensors.empty() || num_classes <= 0 || model_w <= 0 || model_h <= 0 || scales == nullptr) {
        return false;
    }
    struct Cell {
        int index;
        int channels;
        int height;
        int width;
    };
    std::vector<Cell> cells;
    std::vector<std::pair<int, int>> keys;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        int channels = 0;
        int height = 0;
        int width = 0;
        if (!grid_hwc(tensors[i], &channels, &height, &width)) {
            return false;
        }
        cells.push_back(Cell{static_cast<int>(i), channels, height, width});
        bool seen = false;
        for (std::size_t k = 0; k < keys.size(); ++k) {
            if (keys[k].first == height && keys[k].second == width) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            keys.push_back(std::make_pair(height, width));
        }
    }
    if (keys.empty() || keys.size() > 3) {
        return false;
    }
    scales->clear();
    for (std::size_t k = 0; k < keys.size(); ++k) {
        SplitScale scale;
        scale.height = keys[k].first;
        scale.width = keys[k].second;
        int score_count = 0;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            if (cells[i].height != scale.height || cells[i].width != scale.width) {
                continue;
            }
            if (cells[i].channels == 64) {
                if (scale.box >= 0) {
                    return false;
                }
                scale.box = cells[i].index;
            } else if (cells[i].channels == num_classes) {
                scale.score = cells[i].index;
                ++score_count;
            } else {
                return false;
            }
        }
        if (scale.box < 0 || scale.score < 0 || score_count < 1) {
            return false;
        }
        if (model_w % scale.width != 0 || model_h % scale.height != 0) {
            return false;
        }
        if (model_w / scale.width != model_h / scale.height) {
            return false;
        }
        scales->push_back(scale);
    }
    return !scales->empty();
}

void decode_split_scale(const RawTensor& box, const RawTensor& score, int height, int width, const DecodeParams& params,
                        std::vector<Detection>* dets) {
    if (box.data == nullptr || score.data == nullptr) {
        return;
    }
    const int stride = params.letterbox.model_width / width;
    const int classes = params.num_classes;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float best = -1.f;
            int best_id = -1;
            for (int c = 0; c < classes; ++c) {
                float value = dfl_at(score, c, y, x, classes, height, width);
                if (params.dfl_scores_are_logits) {
                    value = sigmoid(value);
                }
                if (value > best) {
                    best = value;
                    best_id = c;
                }
            }
            if (best < params.conf_threshold || best_id < 0) {
                continue;
            }
            float dist[4];
            for (int side = 0; side < 4; ++side) {
                float bins[16];
                for (int b = 0; b < 16; ++b) {
                    bins[b] = dfl_at(box, side * 16 + b, y, x, 64, height, width);
                }
                dist[side] = dfl_expect(bins, 16);
            }
            const float cx = static_cast<float>(x) + 0.5f;
            const float cy = static_cast<float>(y) + 0.5f;
            const float s = static_cast<float>(stride);
            Detection det;
            det.score = best;
            det.class_id = best_id;
            map_box((cx - dist[0]) * s, (cy - dist[1]) * s, (cx + dist[2]) * s, (cy + dist[3]) * s, params.letterbox,
                    &det);
            if (det.w >= 1.f && det.h >= 1.f) {
                dets->push_back(det);
            }
        }
    }
}

}  // namespace

const char* coco_name(int class_id) {
    static const char* kNames[80] = {
        "person",        "bicycle",      "car",           "motorcycle",   "airplane",
        "bus",           "train",        "truck",         "boat",         "traffic light",
        "fire hydrant",  "stop sign",    "parking meter", "bench",        "bird",
        "cat",           "dog",          "horse",         "sheep",        "cow",
        "elephant",      "bear",         "zebra",         "giraffe",      "backpack",
        "umbrella",      "handbag",      "tie",           "suitcase",     "frisbee",
        "skis",          "snowboard",    "sports ball",   "kite",         "baseball bat",
        "baseball glove","skateboard",   "surfboard",     "tennis racket","bottle",
        "wine glass",    "cup",          "fork",          "knife",        "spoon",
        "bowl",          "banana",       "apple",         "sandwich",     "orange",
        "broccoli",      "carrot",       "hot dog",       "pizza",        "donut",
        "cake",          "chair",        "couch",         "potted plant", "bed",
        "dining table",  "toilet",       "tv",            "laptop",       "mouse",
        "remote",        "keyboard",     "cell phone",    "microwave",    "oven",
        "toaster",       "sink",         "refrigerator",  "book",         "clock",
        "vase",          "scissors",     "teddy bear",    "hair drier",   "toothbrush"};
    if (class_id < 0 || class_id >= 80) {
        return "unknown";
    }
    return kNames[class_id];
}

DecodeResult decode_yolov8(const std::vector<RawTensor>& tensors, const DecodeParams& params) {
    DecodeResult result;
    std::vector<Detection> dets;
    if (tensors.size() == 1) {
        int channels = 0;
        int count = 0;
        bool channel_first = false;
        if (!anchor_shape(tensors[0], params.num_classes, &channels, &count, &channel_first)) {
            return result;
        }
        result.layout_recognized = true;
        decode_anchor(tensors[0], params, &dets);
    } else if (tensors.size() == 3) {
        int parsed = 0;
        for (std::size_t i = 0; i < tensors.size(); ++i) {
            if (decode_dfl_head(tensors[i], params, &dets)) {
                ++parsed;
            }
        }
        if (parsed == 0) {
            return result;
        }
        result.layout_recognized = true;
    } else {
        return result;
    }
    result.detections = nms(std::move(dets), params.nms_threshold, params.max_detections);
    return result;
}

bool yolov3_outputs_recognized(const std::vector<RawTensor>& tensors, int num_classes, int model_width,
                               int model_height) {
    if (tensors.size() < 2 || tensors.size() > 3) {
        return false;
    }
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        YoloV3Head head;
        if (!view_yolov3_head(tensors[i], num_classes, model_width, model_height, &head)) {
            return false;
        }
    }
    return true;
}

DecodeResult decode_yolov3(const std::vector<RawTensor>& tensors, const DecodeParams& params) {
    DecodeResult result;
    if (params.anchors.size() != tensors.size() * 6) {
        return result;
    }
    std::vector<YoloV3Head> heads(tensors.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (!view_yolov3_head(tensors[i], params.num_classes, params.letterbox.model_width,
                              params.letterbox.model_height, &heads[i])) {
            return result;
        }
        heads[i].order = static_cast<int>(i);
    }
    std::sort(heads.begin(), heads.end(), [](const YoloV3Head& a, const YoloV3Head& b) {
        if (a.stride != b.stride) {
            return a.stride > b.stride;
        }
        return a.order < b.order;
    });
    result.layout_recognized = true;
    std::vector<Detection> dets;
    for (std::size_t i = 0; i < heads.size(); ++i) {
        const float* anchor_wh = params.anchors.data() + i * 6;
        decode_anchor_head(tensors[static_cast<std::size_t>(heads[i].order)], heads[i], anchor_wh, params, false,
                           &dets);
    }
    result.detections = nms(std::move(dets), params.nms_threshold, params.max_detections);
    return result;
}

bool yolov5_split_dfl_recognized(const std::vector<RawTensor>& tensors, int num_classes, int model_width,
                                 int model_height) {
    std::vector<SplitScale> scales;
    return match_split_dfl(tensors, num_classes, model_width, model_height, &scales);
}

bool yolov5_outputs_recognized(const std::vector<RawTensor>& tensors, int num_classes, int model_width,
                               int model_height) {
    if (yolov5_split_dfl_recognized(tensors, num_classes, model_width, model_height)) {
        return true;
    }
    return yolov3_outputs_recognized(tensors, num_classes, model_width, model_height);
}

DecodeResult decode_yolov5(const std::vector<RawTensor>& tensors, const DecodeParams& params) {
    DecodeResult result;
    std::vector<SplitScale> scales;
    if (match_split_dfl(tensors, params.num_classes, params.letterbox.model_width, params.letterbox.model_height,
                        &scales)) {
        result.layout_recognized = true;
        std::vector<Detection> dets;
        for (std::size_t i = 0; i < scales.size(); ++i) {
            const SplitScale& scale = scales[i];
            decode_split_scale(tensors[static_cast<std::size_t>(scale.box)],
                               tensors[static_cast<std::size_t>(scale.score)], scale.height, scale.width, params,
                               &dets);
        }
        result.detections = nms(std::move(dets), params.nms_threshold, params.max_detections);
        return result;
    }

    if (params.anchors.size() != tensors.size() * 6) {
        return result;
    }
    std::vector<YoloV3Head> heads(tensors.size());
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        if (!view_yolov3_head(tensors[i], params.num_classes, params.letterbox.model_width,
                              params.letterbox.model_height, &heads[i])) {
            return result;
        }
        heads[i].order = static_cast<int>(i);
    }
    std::sort(heads.begin(), heads.end(), [](const YoloV3Head& a, const YoloV3Head& b) {
        if (a.stride != b.stride) {
            return a.stride > b.stride;
        }
        return a.order < b.order;
    });
    result.layout_recognized = true;
    std::vector<Detection> dets;
    for (std::size_t i = 0; i < heads.size(); ++i) {
        const float* anchor_wh = params.anchors.data() + i * 6;
        decode_anchor_head(tensors[static_cast<std::size_t>(heads[i].order)], heads[i], anchor_wh, params, true,
                           &dets);
    }
    result.detections = nms(std::move(dets), params.nms_threshold, params.max_detections);
    return result;
}

std::vector<Detection> suppress_overlaps(std::vector<Detection> dets, float iou_threshold, int max_det) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });
    std::vector<char> removed(dets.size(), 0);
    std::vector<Detection> kept;
    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) {
            continue;
        }
        Detection chosen = dets[i];
        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j] || dets[j].class_id != chosen.class_id) {
                continue;
            }
            if (!same_tiled_target(chosen, dets[j], iou_threshold)) {
                continue;
            }
            const float chosen_area = chosen.w * chosen.h;
            const float other_area = dets[j].w * dets[j].h;
            // 高分的那个如果只是贴着切片边的残框，留下更大的完整框。
            if (other_area > chosen_area * 1.2f && center_inside(chosen, dets[j])) {
                chosen = dets[j];
            }
            removed[j] = 1;
        }
        kept.push_back(chosen);
        if (static_cast<int>(kept.size()) >= max_det) {
            break;
        }
    }
    return kept;
}

bool plan_model_window(InputFit fit, int src_w, int src_h, int model_w, int model_h, ModelWindow* window,
                       WindowAnchor anchor) {
    if (window == nullptr) {
        return false;
    }
    *window = ModelWindow();
    src_w &= ~1;
    src_h &= ~1;
    if (src_w < 2 || src_h < 2 || model_w < 2 || model_h < 2) {
        return false;
    }

    window->meta.src_width = src_w;
    window->meta.src_height = src_h;
    window->meta.model_width = model_w;
    window->meta.model_height = model_h;

    if (fit == InputFit::ShortSide) {
        // 短边刚好铺满模型，长边在原图上居中裁。裁完再缩放，和先缩放再裁中心是同一块区域。
        const bool wider =
            static_cast<long long>(src_w) * model_h > static_cast<long long>(src_h) * model_w;
        int crop_w = src_w;
        int crop_h = src_h;
        int ox = 0;
        int oy = 0;
        if (wider) {
            crop_w = static_cast<int>((static_cast<long long>(src_h) * model_w) / model_h) & ~1;
            if (crop_w > src_w) {
                crop_w = src_w;
            }
            if (crop_w < 2) {
                return false;
            }
            if (anchor == WindowAnchor::Start) {
                ox = 0;
            } else if (anchor == WindowAnchor::End) {
                ox = (src_w - crop_w) & ~1;
            } else {
                ox = ((src_w - crop_w) / 2) & ~1;
            }
            if (ox < 0) {
                ox = 0;
            }
            if (ox + crop_w > src_w) {
                crop_w = (src_w - ox) & ~1;
            }
        } else {
            crop_h = static_cast<int>((static_cast<long long>(src_w) * model_h) / model_w) & ~1;
            if (crop_h > src_h) {
                crop_h = src_h;
            }
            if (crop_h < 2) {
                return false;
            }
            if (anchor == WindowAnchor::Start) {
                oy = 0;
            } else if (anchor == WindowAnchor::End) {
                oy = (src_h - crop_h) & ~1;
            } else {
                oy = ((src_h - crop_h) / 2) & ~1;
            }
            if (oy < 0) {
                oy = 0;
            }
            if (oy + crop_h > src_h) {
                crop_h = (src_h - oy) & ~1;
            }
        }
        if (crop_w < 2 || crop_h < 2) {
            return false;
        }
        window->src_x = ox;
        window->src_y = oy;
        window->src_w = crop_w;
        window->src_h = crop_h;
        window->dst_x = 0;
        window->dst_y = 0;
        window->dst_w = model_w;
        window->dst_h = model_h;
        window->fill = false;
        window->meta.scale = static_cast<float>(model_w) / static_cast<float>(crop_w);
        window->meta.pad_x = -static_cast<float>(ox) * window->meta.scale;
        window->meta.pad_y = -static_cast<float>(oy) * window->meta.scale;
        return window->meta.scale > 0.f;
    }

    const float scale_w = static_cast<float>(model_w) / static_cast<float>(src_w);
    const float scale_h = static_cast<float>(model_h) / static_cast<float>(src_h);
    const float scale = scale_w < scale_h ? scale_w : scale_h;
    int new_w = static_cast<int>(static_cast<float>(src_w) * scale) & ~1;
    int new_h = static_cast<int>(static_cast<float>(src_h) * scale) & ~1;
    if (new_w < 2) {
        new_w = 2;
    }
    if (new_h < 2) {
        new_h = 2;
    }
    if (new_w > model_w) {
        new_w = model_w;
    }
    if (new_h > model_h) {
        new_h = model_h;
    }
    const int pad_x = ((model_w - new_w) / 2) & ~1;
    const int pad_y = ((model_h - new_h) / 2) & ~1;
    window->src_x = 0;
    window->src_y = 0;
    window->src_w = src_w;
    window->src_h = src_h;
    window->dst_x = pad_x;
    window->dst_y = pad_y;
    window->dst_w = new_w;
    window->dst_h = new_h;
    window->fill = true;
    window->meta.scale = static_cast<float>(new_w) / static_cast<float>(src_w);
    window->meta.pad_x = static_cast<float>(pad_x);
    window->meta.pad_y = static_cast<float>(pad_y);
    return window->meta.scale > 0.f;
}

}  // namespace yolo_rknn
