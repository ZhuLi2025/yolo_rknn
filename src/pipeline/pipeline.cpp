#include "yolo_rknn/pipeline/pipeline.hpp"

#include "yolo_rknn/business/controller.hpp"
#include "yolo_rknn/common/logger.hpp"
#include "yolo_rknn/common/thread_util.hpp"
#include "yolo_rknn/decode/mpp_decoder.hpp"
#include "yolo_rknn/display/drm_display.hpp"
#include "yolo_rknn/infer/helmet_classifier.hpp"
#include "yolo_rknn/infer/yolov8_detector.hpp"
#include "yolo_rknn/preprocess/rga_preprocessor.hpp"
#include "yolo_rknn/record/mp4_recorder.hpp"
#include "yolo_rknn/stream/rtsp_client.hpp"

#include <chrono>
#include <utility>

namespace yolo_rknn {
namespace {

int64_t steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

VideoCodec codec_from_name(const std::string& name, bool* ok) {
    *ok = true;
    if (name == "h264") {
        return VideoCodec::H264;
    }
    if (name == "h265") {
        return VideoCodec::H265;
    }
    *ok = false;
    return VideoCodec::H264;
}

}  // namespace

Pipeline::Pipeline(AppConfig config)
    : config_(std::move(config)),
      frames_(static_cast<std::size_t>(config_.queue_depth)),
      display_frames_(static_cast<std::size_t>(config_.queue_depth)),
      results_(static_cast<std::size_t>(config_.queue_depth)),
      osd_results_(static_cast<std::size_t>(config_.queue_depth)),
      record_samples_(static_cast<std::size_t>(config_.queue_depth)) {}

Pipeline::~Pipeline() { stop(); }

void Pipeline::request_stop() { stop_.store(true); }

void Pipeline::notify_ready() {
    {
        std::lock_guard<std::mutex> lock(gate_mu_);
        ++gate_ready_;
    }
    gate_cv_.notify_all();
}

void Pipeline::notify_fail(const std::string& message) {
    {
        std::lock_guard<std::mutex> lock(gate_mu_);
        ++gate_failed_;
        if (gate_error_.empty()) {
            gate_error_ = message;
        }
    }
    stop_.store(true);
    gate_cv_.notify_all();
    YR_LOGE("%s", message.c_str());
}

bool Pipeline::wait_workers(int timeout_ms, std::string* error) {
    std::unique_lock<std::mutex> lock(gate_mu_);
    const int need = worker_count();
    const bool done = gate_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, need] {
        return gate_ready_ + gate_failed_ >= need;
    });
    if (!done) {
        *error = "worker init timeout";
        return false;
    }
    if (gate_failed_ > 0) {
        *error = gate_error_.empty() ? "worker init failed" : gate_error_;
        return false;
    }
    return true;
}

void Pipeline::stream_main() {
    apply_thread_attr(ThreadAttr{"yr_stream", config_.stream_cpu, config_.stream_priority});
    RtspClient client(config_, &stop_);
    Status st = client.open();
    if (!st.ok()) {
        notify_fail(st.message());
        return;
    }
    bool codec_ok = false;
    const VideoCodec codec = codec_from_name(client.codec_name(), &codec_ok);
    if (!codec_ok) {
        notify_fail("unsupported codec " + client.codec_name());
        return;
    }
    std::shared_ptr<MppDecoder> decoder(new MppDecoder(config_));
    st = decoder->init(codec);
    if (!st.ok()) {
        notify_fail(st.message());
        return;
    }
    {
        std::lock_guard<std::mutex> lock(decoder_mu_);
        decoder_ = decoder;
    }
    notify_ready();

    int64_t last_stat = steady_ms();
    uint64_t packets = 0;
    uint64_t produced = 0;
    int bad = 0;
    while (!stop_.load()) {
        EncodedPacket packet;
        const ReadStatus rs = client.read(packet);
        if (rs == ReadStatus::Stopped) {
            break;
        }
        if (rs == ReadStatus::Eos) {
            std::vector<DecodedFrame> tail;
            st = decoder->flush(tail);
            if (!st.ok()) {
                YR_LOGW("flush: %s", st.message().c_str());
            }
            for (std::size_t i = 0; i < tail.size(); ++i) {
                ++produced;
                if (config_.display_enable) {
                    display_frames_.push(tail[i]);
                }
                frames_.push(std::move(tail[i]));
            }
            YR_LOGI("input reached eos, decoded=%llu", static_cast<unsigned long long>(produced));
            break;
        }
        if (rs == ReadStatus::Error) {
            if (stop_.load()) {
                break;
            }
            YR_LOGW("read failed, reconnect in 1s");
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (stop_.load()) {
                break;
            }
            // 不 reset 解码器。推理线程可能还握着上一帧的 DRM buffer。
            st = client.reopen();
            if (!st.ok()) {
                YR_LOGW("reconnect failed: %s", st.message().c_str());
            }
            continue;
        }

        ++packets;
        std::vector<DecodedFrame> decoded;
        st = decoder->decode(packet, decoded);
        if (!st.ok()) {
            YR_LOGE("decode: %s", st.message().c_str());
            if (++bad >= 30) {
                notify_fail("too many VPU errors");
                break;
            }
            continue;
        }
        bad = 0;
        for (std::size_t i = 0; i < decoded.size(); ++i) {
            ++produced;
            if (config_.display_enable) {
                display_frames_.push(decoded[i]);
            }
            frames_.push(std::move(decoded[i]));
        }
        const int64_t now = steady_ms();
        if (config_.stats_interval_ms > 0 && now - last_stat >= config_.stats_interval_ms) {
            YR_LOGI("stream packets=%llu frames=%llu queue_drop=%llu",
                    static_cast<unsigned long long>(packets), static_cast<unsigned long long>(produced),
                    static_cast<unsigned long long>(frames_.dropped()));
            last_stat = now;
        }
    }
    // 文件结束或读线程自己退出时，通知另外两条线程收工。直播被信号打断时这个标志已经是 true。
    YR_LOGI("stream thread exit decoded=%llu dropped=%llu", static_cast<unsigned long long>(produced),
            static_cast<unsigned long long>(frames_.dropped()));
    stop_.store(true);
}

void Pipeline::infer_main() {
    apply_thread_attr(ThreadAttr{"yr_infer", config_.infer_cpu, config_.infer_priority});
    RgaPreprocessor pre(config_);
    Status st = pre.init();
    if (!st.ok()) {
        notify_fail(st.message());
        return;
    }
    std::unique_ptr<YoloV8Detector> detector;
    if (!config_.decode_only) {
        detector.reset(new YoloV8Detector(config_));
        st = detector->init();
        if (!st.ok()) {
            notify_fail(st.message());
            return;
        }
        for (int tile = 0; tile < pre.tile_count(); ++tile) {
            detector->bind_input(tile, pre.buffer_fd(tile), pre.buffer_ptr(tile), pre.buffer_bytes(tile));
        }
    } else {
        YR_LOGI("decode_only: RGA still runs, NPU is skipped");
    }
    std::unique_ptr<HelmetClassifier> helmet;
    if (!config_.decode_only && config_.helmet_enable) {
        helmet.reset(new HelmetClassifier(config_));
        st = helmet->init();
        if (!st.ok()) {
            notify_fail(st.message());
            return;
        }
    }
    notify_ready();

    int64_t last_stat = steady_ms();
    uint64_t inferred = 0;
    // 队列里还有帧就继续做完。停止标志只让拉流线程别再塞新帧，不把已经解出来的 NV12 扔掉。
    while (true) {
        DecodedFrame frame;
        if (!frames_.pop(frame)) {
            break;
        }
        const auto t0 = std::chrono::steady_clock::now();
        PreprocessResult tiles[2];
        int ntiles = 2;
        st = pre.process(frame, tiles, &ntiles);
        const auto t1 = std::chrono::steady_clock::now();
        if (!st.ok()) {
            YR_LOGW("%s", st.message().c_str());
            pre.finish_cpu_read();
            frame.retain.reset();
            frame.fd = -1;
            continue;
        }

        InferenceResult result;
        result.seq = frame.seq;
        result.pts_us = frame.pts_us;
        result.src_width = frame.width;
        result.src_height = frame.height;
        result.preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (detector) {
            st = detector->infer(tiles, ntiles, result);
        }
        if (st.ok() && helmet) {
            const auto c0 = std::chrono::steady_clock::now();
            const Status hs = helmet->classify(frame, result.detections);
            const auto c1 = std::chrono::steady_clock::now();
            result.infer_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
            if (!hs.ok()) {
                YR_LOGW("%s", hs.message().c_str());
            }
        }
        // 录制要读这块 NV12，先把引用交给录制队列，再还给 VPU。
        if (st.ok() && config_.record_enable) {
            RecordSample sample;
            sample.frame = frame;
            sample.result = result;
            record_samples_.push(std::move(sample));
        }
        // 分类要读这块 NV12。RGA 返回之后再还给 VPU。
        frame.retain.reset();
        frame.fd = -1;
        pre.finish_cpu_read();
        if (detector && !st.ok()) {
            YR_LOGW("%s", st.message().c_str());
            continue;
        }
        ++inferred;
        if (config_.display_enable) {
            osd_results_.push(result);
        }
        results_.push(std::move(result));

        const int64_t now = steady_ms();
        if (config_.stats_interval_ms > 0 && now - last_stat >= config_.stats_interval_ms) {
            YR_LOGI("infer done=%llu dropped_frames=%llu", static_cast<unsigned long long>(inferred),
                    static_cast<unsigned long long>(frames_.dropped()));
            last_stat = now;
        }
    }
}

void Pipeline::display_main() {
    apply_thread_attr(ThreadAttr{"yr_disp", config_.display_cpu, config_.display_priority});
    DrmDisplay display(config_);
    Status st = display.init();
    if (!st.ok()) {
        notify_fail(st.message());
        return;
    }
    notify_ready();

    while (true) {
        DecodedFrame frame;
        if (!display_frames_.pop_for(frame, 30)) {
            if (display_frames_.closed()) {
                break;
            }
        } else {
            st = display.present(frame);
            if (!st.ok()) {
                YR_LOGW("%s", st.message().c_str());
            }
        }
        InferenceResult overlay;
        while (osd_results_.pop_for(overlay, 0)) {
            display.update_osd(overlay);
        }
    }
}

void Pipeline::record_main() {
    apply_thread_attr(ThreadAttr{"yr_rec", 2, 10});
    Mp4Recorder recorder(config_);
    Status st = recorder.prepare();
    if (!st.ok()) {
        notify_fail(st.message());
        return;
    }
    notify_ready();

    while (true) {
        RecordSample sample;
        if (!record_samples_.pop(sample)) {
            break;
        }
        st = recorder.write(sample);
        if (!st.ok()) {
            YR_LOGE("%s", st.message().c_str());
            stop_.store(true);
            break;
        }
    }
    YR_LOGI("record thread exit frames=%llu dropped=%llu", static_cast<unsigned long long>(recorder.frames()),
            static_cast<unsigned long long>(record_samples_.dropped()));
}

void Pipeline::business_main() {
    apply_thread_attr(ThreadAttr{"yr_biz", config_.business_cpu, config_.business_priority});
    Controller controller(config_);
    notify_ready();
    while (!stop_.load()) {
        InferenceResult result;
        if (results_.pop_for(result, 50)) {
            controller.on_inference(result);
        }
        if (stop_.load()) {
            break;
        }
        controller.poll();
    }
}

int Pipeline::worker_count() const {
    int count = 3;
    if (config_.display_enable) {
        ++count;
    }
    if (config_.record_enable) {
        ++count;
    }
    return count;
}

Status Pipeline::start() {
    if (stream_thread_.joinable() || infer_thread_.joinable() || display_thread_.joinable() ||
        record_thread_.joinable() || business_thread_.joinable()) {
        return Status::Fail("pipeline already started");
    }
    stop_.store(false);
    stream_thread_ = std::thread(&Pipeline::stream_main, this);
    infer_thread_ = std::thread(&Pipeline::infer_main, this);
    if (config_.display_enable) {
        display_thread_ = std::thread(&Pipeline::display_main, this);
    }
    if (config_.record_enable) {
        record_thread_ = std::thread(&Pipeline::record_main, this);
    }
    business_thread_ = std::thread(&Pipeline::business_main, this);

    std::string error;
    if (!wait_workers(30000, &error)) {
        stop();
        return Status::Fail(error);
    }
    YR_LOGI("pipeline running");
    return Status::Ok();
}

void Pipeline::wait() {
    while (!stop_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    stop();
}

void Pipeline::stop() {
    std::lock_guard<std::mutex> life(life_mu_);
    stop_.store(true);
    if (stream_thread_.joinable()) {
        stream_thread_.join();
    }
    frames_.close();
    display_frames_.close();
    if (infer_thread_.joinable()) {
        infer_thread_.join();
    }
    if (display_thread_.joinable()) {
        display_thread_.join();
    }
    // 推理已经不再往录制队列塞帧，排空后才能拆解码器。
    record_samples_.close();
    if (record_thread_.joinable()) {
        record_thread_.join();
    }
    // 写失败时队列里可能还剩着帧。先丢掉，再拆解码器。
    record_samples_.clear();
    // 推理、显示和录制都已经放开 NV12，再拆解码器。
    frames_.clear();
    display_frames_.clear();
    {
        std::lock_guard<std::mutex> lock(decoder_mu_);
        decoder_.reset();
    }
    results_.close();
    osd_results_.close();
    if (business_thread_.joinable()) {
        business_thread_.join();
    }
    stop_called_ = true;
}

}  // namespace yolo_rknn
