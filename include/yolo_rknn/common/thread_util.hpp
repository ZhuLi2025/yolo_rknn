/**
 * @file thread_util.hpp
 * @brief 给工作线程命名、绑核、提升调度优先级。
 *
 * RK3399Pro 的拓扑（本机）：
 *   cpu0–cpu3  Cortex-A53
 *   cpu4–cpu5  Cortex-A72
 * 绑核本身通常不需要 root；SCHED_FIFO 需要 CAP_SYS_NICE。失败只告警，不让进程退出。
 */
#pragma once

namespace yolo_rknn {

struct ThreadAttr {
    const char* name;       // 最长 15 个字符，超出会被截断
    int cpu;                // 负值表示不绑核
    int fifo_priority;      // 1–99 尝试 SCHED_FIFO；0 表示保持默认调度
};

void apply_thread_attr(const ThreadAttr& attr);

}  // namespace yolo_rknn
