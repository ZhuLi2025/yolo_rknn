/**
 * @file app_config.hpp
 * @brief 纯文本 key=value 配置。不引入 yaml/json 库，板子上少一个依赖就少一种运行时故障。
 */
#pragma once

#include "yolo_rknn/common/status.hpp"

#include <string>
#include <vector>

namespace yolo_rknn {

struct AppConfig {
    // ---- 拉流 ----
    std::string url;
    std::string transport = "tcp";
    bool allow_udp = false;
    int max_delay_us = 100000;          // 解复用抖动缓冲上限，微秒。0 会让部分流起不来
    int buffer_size = 262144;           // 协议层接收缓冲，字节。故意收小，避免堆延迟
    int reorder_queue_size = 0;         // RTSP 乱序重排队列，实时流设 0
    int stimeout_us = 5000000;          // 套接字阻塞超时
    int probe_size = 32768;
    int analyze_duration_us = 100000;

    // ---- 模型 ----
    // yolov8：单输出锚点头，或三个 DFL 头。
    // yolov5：拆开的 DFL 头，或经典 3 个 anchor 头。yolov3：darknet 多尺度 anchor 头。
    std::string model_type = "yolov5";
    std::string model_path;
    // RK3588 三核 NPU。取值 auto / 0 / 1 / 2 / 0_1 / 0_1_2。
    std::string npu_cores = "0_1_2";
    int model_width = 640;
    int model_height = 640;
    // letterbox：整图缩进并填灰。short_side：短边缩放到模型边，长边裁掉。
    std::string preprocess = "letterbox";
    // 1：一个 context，核掩码用 npu_cores。2：仅 short_side 时左右两张并行，分别绑 core0 和 core1。
    int batch_size = 1;
    // 仅 yolov3。每两个数是一个 anchor 的宽高，单位是模型输入像素。
    // 按 stride 从大到小，每个 stride 三个 anchor。空则按输出头个数用 COCO 默认值。
    std::vector<float> anchors;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int num_classes = 80;
    int max_detections = 64;
    int target_class = -1;              // 负值表示不按类别过滤
    bool anchor_scores_are_probabilities = true;
    bool dfl_scores_are_logits = true;
    bool decode_only = false;           // 只跑拉流+硬解+RGA，不加载 NPU 模型
    // 每个检测框再跑一次二分类。关闭时 OSD 仍用检测类别名。
    bool helmet_enable = false;
    std::string helmet_model;

    // ---- 线程 ----
    int queue_depth = 2;
    // 拉流和推理分在两个 A76 簇上，避免抢同一块 L2。业务和显示放在 A55。
    int stream_cpu = 6;                 // A76，2.4 GHz
    int infer_cpu = 4;                  // A76，2.35 GHz
    int business_cpu = 0;               // A55
    int stream_priority = 50;
    int infer_priority = 30;
    int business_priority = 20;

    // ---- 显示 ----
    // 接上 HDMI 后再打开。没接显示器时打开会导致启动失败。
    bool display_enable = false;
    std::string display_device = "/dev/dri/card0";
    int display_cpu = 1;
    int display_priority = 15;

    // ---- 录制 ----
    // 原图叠上 OSD 后写成 H.264 MP4。不依赖 display_enable。
    bool record_enable = false;
    std::string record_dir = "recordings";

    // ---- 解码 ----
    bool low_latency_decode = true;
    int decoder_buf_count = 12;

    // ---- 业务 ----
    float target_bbox_ratio = 0.40f;    // 希望目标框高度占画面高度的比例
    float zoom_deadband = 0.20f;        // 相对死区，避免镜头在阈值附近来回抽动
    float reference_height_m = 10.f;
    float sensor_height_m = 10.f;       // 没有传感器设备时使用的常值
    int zoom_cooldown_ms = 400;
    std::string sensor_device;          // 空：使用 sensor_height_m。文件或 tty：读一行浮点
    std::string camera_device;          // 空：只把 VISCA 指令打到日志
    int camera_baud = 9600;

    std::string log_level = "info";
    int stats_interval_ms = 2000;

    Status load(const std::string& path);
    Status validate() const;
};

}  // namespace yolo_rknn
