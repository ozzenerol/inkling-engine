// OpenAI-compatible API on top of EngineIface: /v1/chat/completions (stream and not), /v1/models,
// /health, /tokenize, /v1/cache. Owner: C.
#pragma once
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine_iface.h"
#include "http.h"
#include "pyjson.h"
#include "prompt_cache.h"
#include "tokenizer/tokenizer.h"

namespace ink {

struct ApiOptions {
  std::string model_id;              // what /v1/models reports (the model dir path)
  int64_t max_context = 131072;
  int default_max_tokens = 16384;
  float default_temperature = 1.0f;  // generation_config.json
  float default_top_p = 1.0f;
  int default_top_k = 0;
  pyjson::Value default_effort;        // null = template default (0.9)
  int64_t prefill_chunk = 2048;      // the engine's prefill chunk (EngineOptions.prefill_chunk)
  int64_t shared_prefix_min = 256;   // see PromptCache::set_shared_prefix_min
  bool splice = true;                // reuse generated token ids for assistant turns sent back
  bool keepalive_comments = true;    // SSE ": prefill" comments during long prefills
  bool log = true;
  std::string dump_dir;              // if set, every chat request body is written there (debugging)
};

class Api {
 public:
  Api(const Tokenizer& tok, EngineIface& eng, ApiOptions o);
  void handle(const http::Request& req, http::Response& res);

  // Prompt token ids for a (canonicalized) request; public for tests.
  std::vector<int32_t> prompt_ids(const pyjson::Value& messages, const pyjson::Value& tools, const pyjson::Value& effort,
                                  int64_t* spliced_tokens = nullptr);

 private:
  void chat(const http::Request& req, http::Response& res);
  void remember_turn(const std::string& key, std::vector<int32_t> ids);
  static std::string turn_key(const std::string& reasoning, const std::string& content,
                              const std::vector<std::pair<std::string, pyjson::Value>>& calls);
  static bool message_key(const pyjson::Value& msg, std::string& key);

  const Tokenizer& tok_;
  EngineIface& eng_;
  ApiOptions o_;
  PromptCache cache_;
  std::vector<int32_t> stops_;
  std::mutex mu_;  // one generation at a time
  std::mutex turns_mu_;
  std::unordered_map<std::string, std::vector<int32_t>> turns_;
  std::deque<std::string> turn_order_;
  // stats
  int64_t n_requests_ = 0, total_prompt_ = 0, total_reused_ = 0, total_spliced_ = 0;
};

}  // namespace ink
