#include "yolo_rknn/preprocess/rga_lock.hpp"

#include <mutex>

namespace yolo_rknn {

namespace {
std::mutex g_rga_mu;
}

RgaApiLock::RgaApiLock() { g_rga_mu.lock(); }

RgaApiLock::~RgaApiLock() { g_rga_mu.unlock(); }

}  // namespace yolo_rknn
