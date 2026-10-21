/**
 * @file rga_preprocessor.hpp
 * @brief 用 RGA 把 NV12 dma-buf 缩放、填边、转成模型需要的 RGB888。
 *
 * 目标缓冲是 DRM CMA 或 dma-heap 的物理连续 dma-buf。CPU 不读 NV12，也不自己做 resize / cvtColor。
 * 缓冲优先申请成非 cacheable，这样 RKNN2 可以直接导入 fd。
 * 只有退回 cacheable 时，process() 才会进入 CPU 读同步，调用方必须 finish_cpu_read()。
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
    int fd = -1;
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

    int buffer_fd() const;
    void* buffer_ptr() const;
    std::size_t buffer_bytes() const;

private:
    struct Impl;
    AppConfig config_;
    Impl* impl_ = nullptr;
};

}  // namespace yolo_rknn
