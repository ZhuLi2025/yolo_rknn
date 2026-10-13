/**
 * @file decoded_frame.hpp
 * @brief 一帧硬解结果。对外只暴露 dma-buf 的 fd 和几何信息，不暴露可被 CPU 遍历的亮度平面。
 *
 * retain 持有 MPP 的 MppFrame。MPP 在 frame 销毁时才把 NV12 buffer 还回 pool。
 * 推理线程用完 RGA 之前不能丢掉 retain，否则 VPU 会把这块内存拿去解下一帧。
 */
#pragma once

#include <cstdint>
#include <memory>

namespace yolo_rknn {

struct DecodedFrame {
    int fd = -1;
    int width = 0;
    int height = 0;
    int hor_stride = 0;
    int ver_stride = 0;
    std::int64_t pts_us = -1;
    std::uint64_t seq = 0;

    // 定义在 mpp_decoder.cpp。用 shared_ptr 是为了让队列在其他编译单元里销毁帧时，
    // 不必看到 MppFrame 的完整类型（deleter 在创建时已经类型擦除）。
    struct Retain;
    std::shared_ptr<Retain> retain;
};

}  // namespace yolo_rknn
