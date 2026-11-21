// Owner: A. Persistent thread pool with parallel_for (CPU reference backend).
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ink {

class ThreadPool {
 public:
  static ThreadPool& get();
  explicit ThreadPool(int n);
  ~ThreadPool();
  int size() const { return int(workers_.size()) + 1; }
  // Calls f(begin, end) on chunks of [0, n). Blocks until done. Not reentrant.
  void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& f, int64_t min_chunk = 1);

 private:
  void run(int id);
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int64_t, int64_t)>* job_ = nullptr;
  int64_t n_ = 0, chunk_ = 1;
  std::atomic<int64_t> next_{0};
  int active_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
};

inline void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& f, int64_t min_chunk = 1) {
  ThreadPool::get().parallel_for(n, f, min_chunk);
}

}  // namespace ink
