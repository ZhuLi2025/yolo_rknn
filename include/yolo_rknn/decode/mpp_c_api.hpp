/**
 * @file mpp_c_api.hpp
 * @brief Rockchip MPP 是 C 接口。集中包一层，避免每个 cpp 自己处理 extern "C"。
 *
 * 包含本头之前可以先 #define MODULE_TAG "模块名"，MPP 内部日志会带上这个标签。
 */
#pragma once

#ifndef MODULE_TAG
#define MODULE_TAG "yolo_rknn"
#endif

extern "C" {
#include "rk_mpi.h"
#include "mpp_buffer.h"
#include "mpp_frame.h"
#include "mpp_packet.h"
}
