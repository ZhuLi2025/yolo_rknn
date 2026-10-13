/**
 * @file dma_buf.hpp
 * @brief 在 CPU 读取 RGA 写过的 dma-buf 之前，通知内核把对应 cache 作废。
 *
 * RGA 走 DMA 写内存。若这块 DRM buffer 是 cacheable 的，CPU 仍可能读到旧 cache。
 * RKNN 1.7 的 rknn_inputs_set 会从用户态指针把张量拷进 NPU，这次拷贝必须看到 RGA 的新数据。
 */
#pragma once

namespace yolo_rknn {

// 返回 false 表示内核不接受这条 ioctl（例如不是 dma-buf）。调用方应记录一次后继续。
bool dma_buf_begin_cpu_read(int fd);
bool dma_buf_end_cpu_read(int fd);
// RGA 写完后再让 CPU 改 OSD。START 丢掉旧 cache，END 把 CPU 画的框刷给显示控制器。
bool dma_buf_begin_cpu_write(int fd);
bool dma_buf_end_cpu_write(int fd);

}  // namespace yolo_rknn
