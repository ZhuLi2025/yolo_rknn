/**
 * @file rga_lock.hpp
 * @brief librga 只有一个设备节点，推理、显示和录制不能同时 ioctl。
 */
#pragma once

namespace yolo_rknn {

class RgaApiLock {
public:
    RgaApiLock();
    ~RgaApiLock();

    RgaApiLock(const RgaApiLock&) = delete;
    RgaApiLock& operator=(const RgaApiLock&) = delete;
};

}  // namespace yolo_rknn
