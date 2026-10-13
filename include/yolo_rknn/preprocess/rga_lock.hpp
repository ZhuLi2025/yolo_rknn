/**
 * @file rga_lock.hpp
 * @brief librga 3.02 只有一个设备节点，推理线程和显示线程不能同时 ioctl。
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
