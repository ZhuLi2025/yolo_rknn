/**
 * @file logger.hpp
 * @brief 线程安全的进程内日志。三个工作线程都会写，必须加锁。
 */
#pragma once

#include <string>

namespace yolo_rknn {

enum class LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3
};

class Logger {
public:
    static Logger& instance();

    void set_level(LogLevel level);
    LogLevel level() const;
    void logf(LogLevel level, const char* file, int line, const char* fmt, ...);

private:
    Logger() = default;
    LogLevel level_ = LogLevel::Info;
};

LogLevel parse_log_level(const std::string& text);

}  // namespace yolo_rknn

#define YR_LOG(level, ...) \
    ::yolo_rknn::Logger::instance().logf((level), __FILE__, __LINE__, __VA_ARGS__)
#define YR_LOGD(...) YR_LOG(::yolo_rknn::LogLevel::Debug, __VA_ARGS__)
#define YR_LOGI(...) YR_LOG(::yolo_rknn::LogLevel::Info, __VA_ARGS__)
#define YR_LOGW(...) YR_LOG(::yolo_rknn::LogLevel::Warn, __VA_ARGS__)
#define YR_LOGE(...) YR_LOG(::yolo_rknn::LogLevel::Error, __VA_ARGS__)
