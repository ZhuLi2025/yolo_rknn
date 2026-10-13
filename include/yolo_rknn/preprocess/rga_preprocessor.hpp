/**
 * @file rga_preprocessor.hpp
 * @brief 用 RGA 把 NV12 dma-buf 缩放、填边、转成模型需要的 RGB888。
 *
 * 目标缓冲是 DRM CMA（物理连续）。CPU 不读 NV12，也不自己做 resize / cvtColor。
 * process() 返回后目标缓冲处于 “CPU 可读” 同步区间，调用方拷完张量后必须 finish_cpu_read()。
 */
#pragma once

#include "yolo_rknn/common/app_config.hpp"
#include "yolo_rknn/common/status.hpp"
#include "yolo_rknn/decode/decoded_frame.hpp"
#include "yolo_rknn/infer/postprocess.hpp"

#include <cstddef>
#include <cstdint>

namespace yolo_rknn {

struct PreprocessResult {
    const std::uint8_t* rgb = nullptr;
    std::size_t bytes = 0;
    LetterboxMeta meta;
};

class RgaPreprocessor {
public:
    explicit RgaPreprocessor(const AppConfig& config);
    ~RgaPreprocessor();

    RgaPreprocessor(const RgaPreprocessor&) = delete;
    RgaPreprocessor& operator=(const RgaPreprocessor&) = delete;

    Status init();
    Status process(const DecodedFrame& frame, PreprocessResult& out);
    void finish_cpu_read();

private:
    struct Impl;
    AppConfig config_;
    Impl* impl_ = nullptr;
};

}  // namespace yolo_rknn
