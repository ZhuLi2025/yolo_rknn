#include "yolo_rknn/common/dma_buf.hpp"

#include <sys/ioctl.h>
#include <cstdint>

namespace yolo_rknn {
namespace {

// 与内核 include/uapi/linux/dma-buf.h 一致。不直接包含那个头，避免和老 4.4 头文件打架。
struct DmaBufSync {
    std::uint64_t flags;
};

constexpr std::uint64_t kDmaBufSyncRead = 1ull << 0;
constexpr std::uint64_t kDmaBufSyncWrite = 2ull << 0;
constexpr std::uint64_t kDmaBufSyncEnd = 1ull << 2;
constexpr unsigned long kDmaBufIoctlSync = _IOW('b', 0, DmaBufSync);

bool sync_ioctl(int fd, std::uint64_t flags) {
    if (fd < 0) {
        return false;
    }
    DmaBufSync sync;
    sync.flags = flags;
    return ::ioctl(fd, kDmaBufIoctlSync, &sync) == 0;
}

}  // namespace

bool dma_buf_begin_cpu_read(int fd) { return sync_ioctl(fd, kDmaBufSyncRead); }

bool dma_buf_end_cpu_read(int fd) { return sync_ioctl(fd, kDmaBufSyncRead | kDmaBufSyncEnd); }

bool dma_buf_begin_cpu_write(int fd) { return sync_ioctl(fd, kDmaBufSyncWrite); }

bool dma_buf_end_cpu_write(int fd) { return sync_ioctl(fd, kDmaBufSyncWrite | kDmaBufSyncEnd); }

}  // namespace yolo_rknn
