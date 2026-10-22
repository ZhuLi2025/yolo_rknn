#include "yolo_rknn/common/app_config.hpp"

#include "yolo_rknn/common/logger.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace yolo_rknn {
namespace {

std::string trim(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool parse_float(const std::string& text, float* out);

bool parse_anchors(const std::string& text, std::vector<float>* out) {
    out->clear();
    if (text.empty()) {
        return true;
    }
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            return false;
        }
        float value = 0.f;
        if (!parse_float(item, &value) || value <= 0.f) {
            return false;
        }
        out->push_back(value);
    }
    // 每个检测头 3 个 anchor，每个 anchor 两个数。
    return !out->empty() && (out->size() % 6) == 0;
}

bool parse_bool(const std::string& text, bool* out) {
    if (text == "1" || text == "true" || text == "yes" || text == "on") {
        *out = true;
        return true;
    }
    if (text == "0" || text == "false" || text == "no" || text == "off") {
        *out = false;
        return true;
    }
    return false;
}

bool parse_int(const std::string& text, int* out) {
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

bool parse_float(const std::string& text, float* out) {
    char* end = nullptr;
    const float value = std::strtof(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    *out = value;
    return true;
}

Status assign(AppConfig* cfg, const std::string& key, const std::string& value, int line) {
    auto bad = [&](const char* what) {
        std::ostringstream os;
        os << "config line " << line << " key " << key << ": " << what;
        return Status::Fail(os.str());
    };

    if (key == "url") {
        cfg->url = value;
    } else if (key == "transport") {
        cfg->transport = value;
    } else if (key == "allow_udp") {
        if (!parse_bool(value, &cfg->allow_udp)) return bad("expect bool");
    } else if (key == "max_delay_us") {
        if (!parse_int(value, &cfg->max_delay_us)) return bad("expect int");
    } else if (key == "buffer_size") {
        if (!parse_int(value, &cfg->buffer_size)) return bad("expect int");
    } else if (key == "reorder_queue_size") {
        if (!parse_int(value, &cfg->reorder_queue_size)) return bad("expect int");
    } else if (key == "stimeout_us") {
        if (!parse_int(value, &cfg->stimeout_us)) return bad("expect int");
    } else if (key == "probe_size") {
        if (!parse_int(value, &cfg->probe_size)) return bad("expect int");
    } else if (key == "analyze_duration_us") {
        if (!parse_int(value, &cfg->analyze_duration_us)) return bad("expect int");
    } else if (key == "model_type") {
        if (value != "yolov8" && value != "yolov5" && value != "yolov3") return bad("expect yolov8, yolov5 or yolov3");
        cfg->model_type = value;
    } else if (key == "model_path") {
        cfg->model_path = value;
    } else if (key == "npu_cores") {
        cfg->npu_cores = value;
    } else if (key == "anchors") {
        if (!parse_anchors(value, &cfg->anchors)) return bad("expect 6*N positive floats, 3 anchors per head");
    } else if (key == "model_width") {
        if (!parse_int(value, &cfg->model_width)) return bad("expect int");
    } else if (key == "model_height") {
        if (!parse_int(value, &cfg->model_height)) return bad("expect int");
    } else if (key == "conf_threshold") {
        if (!parse_float(value, &cfg->conf_threshold)) return bad("expect float");
    } else if (key == "nms_threshold") {
        if (!parse_float(value, &cfg->nms_threshold)) return bad("expect float");
    } else if (key == "num_classes") {
        if (!parse_int(value, &cfg->num_classes)) return bad("expect int");
    } else if (key == "max_detections") {
        if (!parse_int(value, &cfg->max_detections)) return bad("expect int");
    } else if (key == "target_class") {
        if (!parse_int(value, &cfg->target_class)) return bad("expect int");
    } else if (key == "anchor_scores_are_probabilities") {
        if (!parse_bool(value, &cfg->anchor_scores_are_probabilities)) return bad("expect bool");
    } else if (key == "dfl_scores_are_logits") {
        if (!parse_bool(value, &cfg->dfl_scores_are_logits)) return bad("expect bool");
    } else if (key == "decode_only") {
        if (!parse_bool(value, &cfg->decode_only)) return bad("expect bool");
    } else if (key == "run_mode") {
        cfg->decode_only = (value == "decode_only");
        if (value != "decode_only" && value != "full") return bad("expect full or decode_only");
    } else if (key == "queue_depth") {
        if (!parse_int(value, &cfg->queue_depth)) return bad("expect int");
    } else if (key == "stream_cpu") {
        if (!parse_int(value, &cfg->stream_cpu)) return bad("expect int");
    } else if (key == "infer_cpu") {
        if (!parse_int(value, &cfg->infer_cpu)) return bad("expect int");
    } else if (key == "business_cpu") {
        if (!parse_int(value, &cfg->business_cpu)) return bad("expect int");
    } else if (key == "stream_priority") {
        if (!parse_int(value, &cfg->stream_priority)) return bad("expect int");
    } else if (key == "infer_priority") {
        if (!parse_int(value, &cfg->infer_priority)) return bad("expect int");
    } else if (key == "display_enable") {
        if (!parse_bool(value, &cfg->display_enable)) return bad("expect bool");
    } else if (key == "display_device") {
        cfg->display_device = value;
    } else if (key == "display_cpu") {
        if (!parse_int(value, &cfg->display_cpu)) return bad("expect int");
    } else if (key == "display_priority") {
        if (!parse_int(value, &cfg->display_priority)) return bad("expect int");
    } else if (key == "business_priority") {
        if (!parse_int(value, &cfg->business_priority)) return bad("expect int");
    } else if (key == "low_latency_decode") {
        if (!parse_bool(value, &cfg->low_latency_decode)) return bad("expect bool");
    } else if (key == "decoder_buf_count") {
        if (!parse_int(value, &cfg->decoder_buf_count)) return bad("expect int");
    } else if (key == "target_bbox_ratio") {
        if (!parse_float(value, &cfg->target_bbox_ratio)) return bad("expect float");
    } else if (key == "zoom_deadband") {
        if (!parse_float(value, &cfg->zoom_deadband)) return bad("expect float");
    } else if (key == "reference_height_m") {
        if (!parse_float(value, &cfg->reference_height_m)) return bad("expect float");
    } else if (key == "sensor_height_m") {
        if (!parse_float(value, &cfg->sensor_height_m)) return bad("expect float");
    } else if (key == "zoom_cooldown_ms") {
        if (!parse_int(value, &cfg->zoom_cooldown_ms)) return bad("expect int");
    } else if (key == "sensor_device") {
        cfg->sensor_device = value;
    } else if (key == "camera_device") {
        cfg->camera_device = value;
    } else if (key == "camera_baud") {
        if (!parse_int(value, &cfg->camera_baud)) return bad("expect int");
    } else if (key == "log_level") {
        cfg->log_level = value;
    } else if (key == "stats_interval_ms") {
        if (!parse_int(value, &cfg->stats_interval_ms)) return bad("expect int");
    } else {
        YR_LOGW("ignore unknown config key %s at line %d", key.c_str(), line);
    }
    return Status::Ok();
}

bool priority_ok(int value) { return value >= 0 && value <= 99; }

}  // namespace

Status AppConfig::load(const std::string& path) {
    std::ifstream in(path.c_str());
    if (!in) {
        return Status::Fail("cannot open config: " + path);
    }
    std::string line;
    int number = 0;
    while (std::getline(in, line)) {
        ++number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string text = trim(line);
        if (text.empty() || text[0] == '#') {
            continue;
        }
        const std::size_t eq = text.find('=');
        if (eq == std::string::npos) {
            std::ostringstream os;
            os << "config line " << number << " missing '='";
            return Status::Fail(os.str());
        }
        const std::string key = trim(text.substr(0, eq));
        const std::string value = trim(text.substr(eq + 1));
        Status st = assign(this, key, value, number);
        if (!st.ok()) {
            return st;
        }
    }
    return Status::Ok();
}

Status AppConfig::validate() const {
    if (url.empty()) {
        return Status::Fail("url is empty");
    }
    if (transport != "tcp" && transport != "udp") {
        return Status::Fail("transport must be tcp or udp");
    }
    if (transport == "udp" && !allow_udp) {
        return Status::Fail(
            "transport=udp was rejected. LAN packet loss shows up as decoder stalls; "
            "set allow_udp=true only if you accept that");
    }
    if (max_delay_us < 0 || buffer_size < 4096 || stimeout_us < 0) {
        return Status::Fail("max_delay_us / buffer_size / stimeout_us out of range");
    }
    if (model_type != "yolov8" && model_type != "yolov5" && model_type != "yolov3") {
        return Status::Fail("model_type must be yolov8, yolov5 or yolov3");
    }
    if (npu_cores != "auto" && npu_cores != "0" && npu_cores != "1" && npu_cores != "2" && npu_cores != "0_1" &&
        npu_cores != "0_1_2") {
        return Status::Fail("npu_cores must be auto, 0, 1, 2, 0_1 or 0_1_2");
    }
    if (!decode_only && model_path.empty()) {
        return Status::Fail("model_path is empty (or set decode_only=true)");
    }
    if ((model_type == "yolov3" || model_type == "yolov5") && !anchors.empty() && (anchors.size() % 6) != 0) {
        return Status::Fail("anchors must list 3 per head, each as width,height");
    }
    // RK3588 的 RGA3 要求宽高 16 对齐，RGB888 的字节跨度才能被硬件接受。
    if (model_width < 16 || model_height < 16 || (model_width % 16) != 0 || (model_height % 16) != 0) {
        return Status::Fail("model_width and model_height must be positive multiples of 16");
    }
    if (num_classes <= 0 || max_detections <= 0 || queue_depth < 1 || decoder_buf_count < 4) {
        return Status::Fail("num_classes, max_detections, queue_depth or decoder_buf_count is invalid");
    }
    if (!priority_ok(stream_priority) || !priority_ok(infer_priority) || !priority_ok(business_priority) ||
        !priority_ok(display_priority)) {
        return Status::Fail("thread priority must be in 0..99");
    }
    if (display_enable && display_device.empty()) {
        return Status::Fail("display_device is empty");
    }
    if (conf_threshold < 0.f || nms_threshold < 0.f || target_bbox_ratio <= 0.f) {
        return Status::Fail("threshold or target_bbox_ratio is invalid");
    }
    return Status::Ok();
}

}  // namespace yolo_rknn
