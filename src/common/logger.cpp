#include "yolo_rknn/common/logger.hpp"

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>

namespace yolo_rknn {
namespace {

std::mutex g_log_mu;

const char* level_name(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:
            return "D";
        case LogLevel::Info:
            return "I";
        case LogLevel::Warn:
            return "W";
        case LogLevel::Error:
            return "E";
    }
    return "?";
}

const char* basename(const char* path) {
    const char* base = path;
    for (const char* p = path; p && *p; ++p) {
        if (*p == '/') {
            base = p + 1;
        }
    }
    return base ? base : "";
}

}  // namespace

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(g_log_mu);
    level_ = level;
}

LogLevel Logger::level() const {
    std::lock_guard<std::mutex> lock(g_log_mu);
    return level_;
}

void Logger::logf(LogLevel level, const char* file, int line, const char* fmt, ...) {
    {
        std::lock_guard<std::mutex> lock(g_log_mu);
        if (static_cast<int>(level) < static_cast<int>(level_)) {
            return;
        }
    }

    char body[1280];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(body, sizeof(body), fmt, ap);
    va_end(ap);

    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    tm local;
    localtime_r(&ts.tv_sec, &local);
    char stamp[32];
    std::snprintf(stamp, sizeof(stamp), "%02d:%02d:%02d.%03ld", local.tm_hour, local.tm_min,
                  local.tm_sec, ts.tv_nsec / 1000000L);

    std::lock_guard<std::mutex> lock(g_log_mu);
    std::fprintf(stderr, "%s %s %s:%d %s\n", level_name(level), stamp, basename(file), line, body);
    std::fflush(stderr);
}

LogLevel parse_log_level(const std::string& text) {
    if (text == "debug") {
        return LogLevel::Debug;
    }
    if (text == "warn" || text == "warning") {
        return LogLevel::Warn;
    }
    if (text == "error") {
        return LogLevel::Error;
    }
    return LogLevel::Info;
}

}  // namespace yolo_rknn
