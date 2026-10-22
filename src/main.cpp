#include "yolo_rknn/common/logger.hpp"
#include "yolo_rknn/pipeline/pipeline.hpp"

#include <csignal>
#include <fstream>
#include <iostream>
#include <string>

namespace {

yolo_rknn::Pipeline* g_pipeline = nullptr;

void on_signal(int) {
    if (g_pipeline != nullptr) {
        g_pipeline->request_stop();
    }
}

void usage() {
    std::cerr
        << "yolo_rknn " << YR_VERSION << "\n"
        << "用法:\n"
        << "  yolo_rknn [--config config/app.conf] [--url URL] [--model FILE] [--decode-only]\n"
        << "\n"
        << "  --decode-only   只跑拉流、VPU 硬解和 RGA，不加载 NPU 模型\n"
        << "默认会读取当前目录下的 config/app.conf（如果存在）。\n";
}

std::string device_model() {
    std::ifstream in("/proc/device-tree/model");
    std::string text;
    std::getline(in, text);
    while (!text.empty() && (text.back() == '\0' || text.back() == '\n')) {
        text.pop_back();
    }
    return text;
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    std::string url;
    std::string model;
    bool decode_only = false;
    bool config_given = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage();
            return 0;
        }
        if (arg == "--decode-only") {
            decode_only = true;
            continue;
        }
        if (i + 1 >= argc) {
            usage();
            return 2;
        }
        if (arg == "--config") {
            config_path = argv[++i];
            config_given = true;
        } else if (arg == "--url") {
            url = argv[++i];
        } else if (arg == "--model") {
            model = argv[++i];
        } else {
            usage();
            return 2;
        }
    }

    yolo_rknn::AppConfig cfg;
    yolo_rknn::Status st = yolo_rknn::Status::Ok();
    if (config_given) {
        st = cfg.load(config_path);
    } else {
        std::ifstream probe("config/app.conf");
        if (probe.good()) {
            st = cfg.load("config/app.conf");
        }
    }
    if (!st.ok()) {
        std::cerr << st.message() << "\n";
        return 1;
    }
    if (!url.empty()) {
        cfg.url = url;
    }
    if (!model.empty()) {
        cfg.model_path = model;
    }
    if (decode_only) {
        cfg.decode_only = true;
    }
    st = cfg.validate();
    if (!st.ok()) {
        std::cerr << st.message() << "\n";
        return 1;
    }
    yolo_rknn::Logger::instance().set_level(yolo_rknn::parse_log_level(cfg.log_level));

    const std::string board = device_model();
    YR_LOGI("yolo_rknn %s device=%s", YR_VERSION, board.empty() ? "unknown" : board.c_str());
    YR_LOGI("url=%s model=%s type=%s preprocess=%s batch=%d decode_only=%d", cfg.url.c_str(), cfg.model_path.c_str(),
            cfg.model_type.c_str(), cfg.preprocess.c_str(), cfg.batch_size, cfg.decode_only ? 1 : 0);

    yolo_rknn::Pipeline pipeline(cfg);
    g_pipeline = &pipeline;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    st = pipeline.start();
    if (!st.ok()) {
        YR_LOGE("start failed: %s", st.message().c_str());
        g_pipeline = nullptr;
        return 1;
    }
    pipeline.wait();
    g_pipeline = nullptr;
    YR_LOGI("stopped");
    return 0;
}
