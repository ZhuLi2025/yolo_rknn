/**
 * @file status.hpp
 * @brief 跨模块传递成功或失败，不在线程之间抛异常。
 */
#pragma once

#include <string>
#include <utility>

namespace yolo_rknn {

class Status {
public:
    static Status Ok() { return Status(true, std::string()); }
    static Status Fail(std::string message) { return Status(false, std::move(message)); }

    bool ok() const { return ok_; }
    const std::string& message() const { return message_; }

private:
    Status(bool ok, std::string message) : ok_(ok), message_(std::move(message)) {}

    bool ok_;
    std::string message_;
};

}  // namespace yolo_rknn
