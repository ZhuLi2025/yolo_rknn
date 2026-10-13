/**
 * @file pipeline.hpp
 * @brief 线程装配点。显示关闭时三条，打开时四条。线程之间只通过有界队列交换数据，不共享 MPP / RKNN context。
 *
 *   线程 stream    拉流 + VPU 硬解，产出 DecodedFrame（同时进推理队列和显示队列）
 *   线程 infer     RGA 预处理 + NPU 推理，产出 InferenceResult（同时进业务和 OSD）
 *   线程 display   DRM 视频层扫 NV12，OSD 层画框
 *   线程 business  高度、变焦、相机指令
 */
#pragma once

#include "yolo_rknn/common/app_config.hpp"
#include "yolo_rknn/common/bounded_queue.hpp"
#include "yolo_rknn/common/status.hpp"
#include "yolo_rknn/decode/decoded_frame.hpp"
#include "yolo_rknn/infer/detection.hpp"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace yolo_rknn {

class MppDecoder;

class Pipeline {
public:
    explicit Pipeline(AppConfig config);
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    Status start();
    // 信号处理器里只能调用这个：它只写一个 atomic。
    void request_stop();
    void wait();
    void stop();

private:
    void stream_main();
    void infer_main();
    void display_main();
    void business_main();
    void notify_ready();
    void notify_fail(const std::string& message);
    bool wait_workers(int timeout_ms, std::string* error);
    int worker_count() const;

    AppConfig config_;
    std::atomic<bool> stop_{false};
    BoundedQueue<DecodedFrame> frames_;
    BoundedQueue<DecodedFrame> display_frames_;
    BoundedQueue<InferenceResult> results_;
    BoundedQueue<InferenceResult> osd_results_;

    std::thread stream_thread_;
    std::thread infer_thread_;
    std::thread display_thread_;
    std::thread business_thread_;

    std::mutex life_mu_;
    bool stop_called_ = false;

    std::mutex gate_mu_;
    std::condition_variable gate_cv_;
    int gate_ready_ = 0;
    int gate_failed_ = 0;
    std::string gate_error_;

    // 解码器在拉流线程创建，但必须活到推理线程还完最后一帧 NV12。
    std::mutex decoder_mu_;
    std::shared_ptr<MppDecoder> decoder_;
};

}  // namespace yolo_rknn
