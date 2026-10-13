/**
 * @file app_config.hpp
 * @brief 纯文本 key=value 配置。不引入 yaml/json 库，板子上少一个依赖就少一种运行时故障。
 */
#pragma once

#include "yolo_rknn/common/status.hpp"

#include <string>

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
    std::string model_path;
    int model_width = 640;
    int model_height = 640;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int num_classes = 80;
    int max_detections = 64;
    int target_class = -1;              // 负值表示不按类别过滤
    bool anchor_scores_are_probabilities = true;
    bool dfl_scores_are_logits = true;
    bool decode_only = false;           // 只跑拉流+硬解+RGA，不加载 NPU 模型

    // ---- 线程 ----
    int queue_depth = 2;
    int stream_cpu = 4;                 // A72
    int infer_cpu = 5;                  // A72，只负责把张量送进 NPU
    int business_cpu = 0;               // A53
    int stream_priority = 50;
    int infer_priority = 30;
    int business_priority = 20;

    // ---- 显示 ----
    // 接上 HDMI 后再打开。没接显示器时打开会导致启动失败。
    bool display_enable = false;
    std::string display_device = "/dev/dri/card0";
    int display_cpu = 1;
    int display_priority = 15;

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
