#include "core/parallel.h"

#include <algorithm>

namespace ink {

ThreadPool& ThreadPool::get() {
  static ThreadPool pool(std::max(1u, std::thread::hardware_concurrency()));
  return pool;
}

ThreadPool::ThreadPool(int n) {
  for (int i = 1; i < n; ++i) workers_.emplace_back([this, i] { run(i); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> l(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::run(int) {
  uint64_t seen = 0;
  while (true) {
    const std::function<void(int64_t, int64_t)>* job;
    {
      std::unique_lock<std::mutex> l(mu_);
      cv_.wait(l, [&] { return stop_ || gen_ != seen; });
      if (stop_) return;
      seen = gen_;
      job = job_;
    }
    while (true) {
      int64_t b = next_.fetch_add(chunk_);
      if (b >= n_) break;
      (*job)(b, std::min(n_, b + chunk_));
    }
    {
      std::lock_guard<std::mutex> l(mu_);
      if (++active_ == int(workers_.size())) done_cv_.notify_all();
    }
  }
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& f, int64_t min_chunk) {
  if (n <= 0) return;
  int64_t chunk = std::max<int64_t>(min_chunk, (n + size() * 4 - 1) / (size() * 4));
  if (workers_.empty() || n <= chunk) {
    f(0, n);
    return;
  }
  {
    std::lock_guard<std::mutex> l(mu_);
    job_ = &f;
    n_ = n;
    chunk_ = chunk;
    next_ = 0;
    active_ = 0;  // number of workers done with this generation
    ++gen_;
  }
  cv_.notify_all();
  while (true) {
    int64_t b = next_.fetch_add(chunk);
    if (b >= n) break;
    f(b, std::min(n, b + chunk));
  }
  std::unique_lock<std::mutex> l(mu_);
  done_cv_.wait(l, [&] { return active_ == int(workers_.size()); });
  job_ = nullptr;
}

}  // namespace ink
