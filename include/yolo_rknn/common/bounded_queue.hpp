/**
 * @file bounded_queue.hpp
 * @brief 有界队列。默认在队满时丢掉最旧元素，保证拉流线程永远不会被推理线程堵住。
 *
 * 实时视频里堆积旧帧只会增加端到端延迟。工业场景要的是“最新的一帧”，不是“每一帧都处理完”。
 * 元素在锁外析构，避免 NV12 帧归还 MPP buffer 时占着队列锁。
 */
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>

namespace yolo_rknn {

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity)
        : capacity_(capacity == 0 ? 1 : capacity) {}

    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    // 关闭后，正在等待的 pop 会醒来；已在队列里的元素仍然可以取出。
    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool closed() const {
        std::lock_guard<std::mutex> lock(mu_);
        return closed_;
    }

    // 返回 false 表示队列已关闭，元素没有入队。
    bool push(T value) {
        T dropped_item;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (closed_) {
                return false;
            }
            if (q_.size() >= capacity_) {
                dropped_item = std::move(q_.front());
                q_.pop_front();
                dropped_.fetch_add(1, std::memory_order_relaxed);
            }
            q_.push_back(std::move(value));
        }
        cv_.notify_one();
        return true;
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return closed_ || !q_.empty(); });
        if (q_.empty()) {
            return false;
        }
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    // timeout_ms 到期且没有数据时返回 false。队列关闭且已空同样返回 false。
    bool pop_for(T& out, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
            return closed_ || !q_.empty();
        });
        if (q_.empty()) {
            return false;
        }
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    void clear() {
        std::deque<T> discarded;
        {
            std::lock_guard<std::mutex> lock(mu_);
            discarded.swap(q_);
        }
    }

    std::uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return q_.size();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<T> q_;
    std::size_t capacity_;
    bool closed_ = false;
    std::atomic<std::uint64_t> dropped_{0};
};

}  // namespace yolo_rknn
