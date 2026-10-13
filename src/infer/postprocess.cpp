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

void map_box(float x1, float y1, float x2, float y2, const LetterboxMeta& meta, Detection* det) {
    if (meta.scale <= 0.f) {
        return;
    }
    x1 = (x1 - static_cast<float>(meta.pad_x)) / meta.scale;
    y1 = (y1 - static_cast<float>(meta.pad_y)) / meta.scale;
    x2 = (x2 - static_cast<float>(meta.pad_x)) / meta.scale;
    y2 = (y2 - static_cast<float>(meta.pad_y)) / meta.scale;
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

}  // namespace yolo_rknn
