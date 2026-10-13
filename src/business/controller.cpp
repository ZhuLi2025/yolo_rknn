#include "yolo_rknn/business/controller.hpp"

#include "yolo_rknn/common/logger.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

namespace yolo_rknn {
namespace {

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

speed_t baud_flag(int baud) {
    switch (baud) {
        case 19200:
            return B19200;
        case 38400:
            return B38400;
        case 57600:
            return B57600;
        case 115200:
            return B115200;
        default:
            return B9600;
    }
}

bool is_tty_path(const std::string& path) {
    return path.compare(0, 8, "/dev/tty") == 0;
}

const char* action_name(ZoomAction action) {
    switch (action) {
        case ZoomAction::ZoomIn:
            return "zoom-in";
        case ZoomAction::ZoomOut:
            return "zoom-out";
        case ZoomAction::Hold:
            return "hold";
    }
    return "hold";
}

}  // namespace

struct Controller::Impl {
    int camera_fd = -1;
    int sensor_fd = -1;
    std::string sensor_pending;
};

Controller::Controller(const AppConfig& config) : config_(config), impl_(new Impl) {
    if (config_.camera_device.empty()) {
        YR_LOGI("camera control is log-only. Set camera_device to send VISCA");
        return;
    }
    impl_->camera_fd = ::open(config_.camera_device.c_str(), O_RDWR | O_NOCTTY | O_CLOEXEC);
    if (impl_->camera_fd < 0) {
        YR_LOGW("open camera %s failed, VISCA will be logged only", config_.camera_device.c_str());
        return;
    }
    termios tio;
    if (tcgetattr(impl_->camera_fd, &tio) != 0) {
        YR_LOGW("tcgetattr %s failed", config_.camera_device.c_str());
        return;
    }
    cfmakeraw(&tio);
    const speed_t speed = baud_flag(config_.camera_baud);
    cfsetispeed(&tio, speed);
    cfsetospeed(&tio, speed);
    tio.c_cflag |= (CLOCAL | CREAD);
    tio.c_cflag &= ~CSTOPB;
    tio.c_cflag &= ~PARENB;
    tio.c_cflag = (tio.c_cflag & ~CSIZE) | CS8;
    if (tcsetattr(impl_->camera_fd, TCSANOW, &tio) != 0) {
        YR_LOGW("tcsetattr %s failed", config_.camera_device.c_str());
    } else {
        YR_LOGI("VISCA camera %s baud=%d", config_.camera_device.c_str(), config_.camera_baud);
    }
    if (is_tty_path(config_.sensor_device)) {
        impl_->sensor_fd = ::open(config_.sensor_device.c_str(), O_RDONLY | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
        if (impl_->sensor_fd < 0) {
            YR_LOGW("open height sensor %s failed, using sensor_height_m", config_.sensor_device.c_str());
        }
    }
}

Controller::~Controller() {
    if (impl_ != nullptr && impl_->camera_fd >= 0) {
        ::close(impl_->camera_fd);
    }
    if (impl_ != nullptr && impl_->sensor_fd >= 0) {
        ::close(impl_->sensor_fd);
    }
    delete impl_;
    impl_ = nullptr;
}

float Controller::read_height(bool* valid) {
    *valid = false;
    if (config_.sensor_device.empty()) {
        *valid = true;
        return config_.sensor_height_m;
    }
    if (!is_tty_path(config_.sensor_device)) {
        std::ifstream in(config_.sensor_device.c_str());
        float meters = 0.f;
        if (in >> meters) {
            *valid = true;
            return meters;
        }
        return config_.sensor_height_m;
    }

    // 串口传感器约定：一行一个米制浮点数。非阻塞读，凑齐一行再解析。
    if (impl_->sensor_fd < 0) {
        return config_.sensor_height_m;
    }
    char buf[128];
    const ssize_t n = ::read(impl_->sensor_fd, buf, sizeof(buf));
    if (n > 0) {
        impl_->sensor_pending.append(buf, buf + n);
    }
    const std::size_t nl = impl_->sensor_pending.find('\n');
    if (nl == std::string::npos) {
        return config_.sensor_height_m;
    }
    const std::string line = impl_->sensor_pending.substr(0, nl);
    impl_->sensor_pending.erase(0, nl + 1);
    std::istringstream iss(line);
    float meters = 0.f;
    if (iss >> meters) {
        *valid = true;
        return meters;
    }
    return config_.sensor_height_m;
}

void Controller::apply(ZoomAction action) {
    const int64_t now = now_ms();
    const bool stopping = action == ZoomAction::Hold && last_action_ != ZoomAction::Hold;
    if (action == last_action_) {
        return;
    }
    if (!stopping && config_.zoom_cooldown_ms > 0 && (now - last_cmd_ms_) < config_.zoom_cooldown_ms) {
        return;
    }

    // VISCA 标准变焦。地址固定 1。停机指令在动作变回 Hold 时发一次，避免镜头停不下来。
    static const uint8_t kZoomIn[] = {0x81, 0x01, 0x04, 0x07, 0x02, 0xFF};
    static const uint8_t kZoomOut[] = {0x81, 0x01, 0x04, 0x07, 0x03, 0xFF};
    static const uint8_t kZoomStop[] = {0x81, 0x01, 0x04, 0x07, 0x00, 0xFF};
    const uint8_t* cmd = kZoomStop;
    if (action == ZoomAction::ZoomIn) {
        cmd = kZoomIn;
    } else if (action == ZoomAction::ZoomOut) {
        cmd = kZoomOut;
    }

    if (impl_->camera_fd >= 0) {
        const ssize_t n = ::write(impl_->camera_fd, cmd, 6);
        if (n != 6) {
            YR_LOGW("VISCA write failed");
        }
    }
    YR_LOGI("camera %s", action_name(action));
    last_action_ = action;
    last_cmd_ms_ = now;
}

void Controller::on_inference(const InferenceResult& result) {
    latest_ = result;
    has_result_ = true;
}

void Controller::poll() {
    bool height_valid = false;
    const float height = read_height(&height_valid);

    ZoomInput input;
    input.height_valid = height_valid;
    input.height_m = height;
    input.reference_height_m = config_.reference_height_m;
    input.target_ratio = config_.target_bbox_ratio;
    input.deadband = config_.zoom_deadband;

    const Detection* best = nullptr;
    if (has_result_) {
        for (std::size_t i = 0; i < latest_.detections.size(); ++i) {
            const Detection& det = latest_.detections[i];
            if (config_.target_class >= 0 && det.class_id != config_.target_class) {
                continue;
            }
            if (best == nullptr || det.h > best->h) {
                best = &det;
            }
        }
    }
    if (best != nullptr && latest_.src_height > 0) {
        input.has_target = true;
        input.bbox_h_ratio = best->h / static_cast<float>(latest_.src_height);
    }
    apply(decide_zoom(input));

    if (!has_result_) {
        return;
    }
    static int64_t last_log = 0;
    const int64_t now = now_ms();
    if (now - last_log < 1000) {
        return;
    }
    last_log = now;
    if (best != nullptr) {
        YR_LOGI("seq=%llu det=%zu top=%s score=%.2f box=(%.0f,%.0f,%.0f,%.0f) ratio=%.2f height=%.2fm infer=%.1fms",
                static_cast<unsigned long long>(latest_.seq), latest_.detections.size(), coco_name(best->class_id),
                best->score, best->x, best->y, best->w, best->h, input.bbox_h_ratio, height, latest_.infer_ms);
    } else {
        YR_LOGI("seq=%llu det=0 height=%.2fm preprocess=%.1fms infer=%.1fms",
                static_cast<unsigned long long>(latest_.seq), height, latest_.preprocess_ms, latest_.infer_ms);
    }
}

}  // namespace yolo_rknn
