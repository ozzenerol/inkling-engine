#include "prompt_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace ink {

void PromptCache::drop_above(int64_t len) {
  for (auto it = ck_.begin(); it != ck_.end();) {
    if (it->c.len > len) { e_.drop(it->c); it = ck_.erase(it); }
    else ++it;
  }
}

void PromptCache::take_ckpt() {
  int64_t len = (int64_t)toks_.size();
  for (auto& x : ck_)
    if (x.c.len == len) { x.used = ++clock_; return; }
  while ((int)ck_.size() >= max_ckpts_ && !ck_.empty()) {
    auto lru = std::min_element(ck_.begin(), ck_.end(), [](const Entry& a, const Entry& b) { return a.used < b.used; });
    e_.drop(lru->c);
    ck_.erase(lru);
  }
  if (max_ckpts_ <= 0) return;
  Ckpt c = e_.checkpoint();
  if (c.id < 0) return;
  ck_.push_back({c, ++clock_});
}

void PromptCache::invalidate() {
  for (auto& x : ck_) e_.drop(x.c);
  ck_.clear();
  toks_.clear();
  e_.reset();
}

PromptCache::Prep PromptCache::prepare(const std::vector<int32_t>& prompt,
                                       const std::function<void(int64_t, int64_t)>& progress) {
  Prep r;
  const int64_t n = (int64_t)prompt.size();
  if (e_.cached_len() != (int64_t)toks_.size()) {
    std::fprintf(stderr, "[cache] engine has %lld tokens, cache thinks %zu: resetting\n",
                 (long long)e_.cached_len(), toks_.size());
    invalidate();
  }
  int64_t lcp = 0;
  const int64_t m = std::min<int64_t>(n, toks_.size());
  while (lcp < m && prompt[lcp] == toks_[lcp]) lcp++;
  // At least one token must be fed to get logits for the next one.
  const int64_t target = std::min(lcp, n - 1);
  int64_t cur = (int64_t)toks_.size();
  r.how = "hit";
  if (target < cur) {
    if (e_.rewind(target)) {
      cur = target;
      r.how = "rewind";
    } else {
      const Entry* best = nullptr;
      for (auto& x : ck_)
        if (x.c.len <= target && (!best || x.c.len > best->c.len)) best = &x;
      if (best && e_.restore(best->c)) {
        cur = best->c.len;
        const_cast<Entry*>(best)->used = ++clock_;
        r.how = "ckpt";
      } else {
        for (auto& x : ck_) e_.drop(x.c);
        ck_.clear();
        e_.reset();
        cur = 0;
        r.how = "reset";
      }
    }
    toks_.resize(cur);
    drop_above(cur);
  }
  if (cur == 0 && r.how == "hit") r.how = "cold";
  r.reused = cur;
  std::span<const int32_t> rest(prompt.data() + cur, prompt.size() - cur);
  r.prefilled = (int64_t)rest.size();
  auto t0 = std::chrono::steady_clock::now();
  e_.prefill(rest, [&](int64_t done) {
    if (progress) progress(cur + done, n);
  });
  r.prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  toks_ = prompt;
  take_ckpt();
  return r;
}

void PromptCache::after_generate(const std::vector<int32_t>& emitted, bool turn_done) {
  if (!emitted.empty()) toks_.insert(toks_.end(), emitted.begin(), emitted.end() - 1);
  if (e_.cached_len() != (int64_t)toks_.size()) {
    std::fprintf(stderr, "[cache] after generate: engine %lld vs %zu tokens: resetting\n",
                 (long long)e_.cached_len(), toks_.size());
    invalidate();
    return;
  }
  if (turn_done) take_ckpt();
}

}  // namespace ink
