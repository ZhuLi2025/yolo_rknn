#include "yolo_rknn/common/thread_util.hpp"

#include "yolo_rknn/common/logger.hpp"

#include <pthread.h>
#include <sched.h>

#include <cstring>

namespace yolo_rknn {

void apply_thread_attr(const ThreadAttr& attr) {
    if (attr.name && attr.name[0] != '\0') {
        char name[16];
        std::memset(name, 0, sizeof(name));
        std::strncpy(name, attr.name, sizeof(name) - 1);
        int rc = pthread_setname_np(pthread_self(), name);
        if (rc != 0) {
            YR_LOGW("set thread name %s failed: %d", name, rc);
        }
    }

    if (attr.cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(attr.cpu, &set);
        int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        if (rc != 0) {
            YR_LOGW("bind cpu%d failed: %d", attr.cpu, rc);
        } else {
            const char* kind = (attr.cpu >= 4) ? "A72" : "A53";
            YR_LOGI("thread %s bound to cpu%d (%s)", attr.name ? attr.name : "?", attr.cpu, kind);
        }
    }

    if (attr.fifo_priority > 0) {
        int priority = attr.fifo_priority;
        int min_pri = sched_get_priority_min(SCHED_FIFO);
        int max_pri = sched_get_priority_max(SCHED_FIFO);
        if (priority < min_pri) {
            priority = min_pri;
        }
        if (priority > max_pri) {
            priority = max_pri;
        }
        sched_param param;
        std::memset(&param, 0, sizeof(param));
        param.sched_priority = priority;
        int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
        if (rc != 0) {
            YR_LOGW("SCHED_FIFO %d not applied (%d). Need CAP_SYS_NICE or root; "
                    "thread isolation still holds via affinity",
                    priority, rc);
        } else {
            YR_LOGI("thread %s SCHED_FIFO priority %d", attr.name ? attr.name : "?", priority);
        }
    }
}

}  // namespace yolo_rknn
