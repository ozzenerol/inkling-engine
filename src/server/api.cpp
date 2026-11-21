#include "api.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <random>

#include "chat_template.h"
#include "output_parser.h"

namespace ink {
using pyjson::Value;

namespace {

std::string trim(const std::string& s) {
  const char* ws = " \t\n\r\f\v";
  size_t a = s.find_first_not_of(ws);
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(ws);
  return s.substr(a, b - a + 1);
}

std::string error_json(const std::string& msg, const std::string& type = "invalid_request_error") {
  Value e = Value::object();
  Value in = Value::object();
  in.set("message", msg);
  in.set("type", type);
  e.set("error", std::move(in));
  return pyjson::compact(e);
}

std::string new_id() {
  static std::mt19937_64 rng(std::random_device{}());
  static std::mutex mu;
  std::lock_guard<std::mutex> g(mu);
  char b[32];
  std::snprintf(b, sizeof b, "chatcmpl-%016llx", (unsigned long long)rng());
  return b;
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const pyjson::DumpOptions kSorted{",", ":", true, false};

}  // namespace

Api::Api(const Tokenizer& tok, EngineIface& eng, ApiOptions o)
    : tok_(tok), eng_(eng), o_(std::move(o)), cache_(eng), stops_(OutputParser::stop_ids(tok)) {}

std::string Api::turn_key(const std::string& reasoning, const std::string& content,
                          const std::vector<std::pair<std::string, Value>>& calls) {
  std::string k = trim(reasoning);
  k += '\x1f';
  k += trim(content);
  for (auto& [name, args] : calls) {
    k += '\x1e';
    k += name;
    k += '\x1d';
    k += pyjson::dump(args, kSorted);
  }
  return k;
}

bool Api::message_key(const Value& m, std::string& key) {
  const Value& c = m["content"];
  std::string content;
  if (c.is_string()) content = c.s;
  else if (!c.is_null()) return false;
  std::string reasoning = m["reasoning_content"].truthy() ? pyjson::py_str(m["reasoning_content"]) : "";
  std::vector<std::pair<std::string, Value>> calls;
  const Value& tcs = m["tool_calls"];
  if (tcs.is_array()) {
    for (auto& tc : tcs.arr) {
      const Value& fn = tc["function"];
      if (!fn["name"].is_string()) return false;
      const Value& a = fn["arguments"];
      calls.emplace_back(fn["name"].s, a.truthy() ? a : Value::object());
    }
  }
  key = turn_key(reasoning, content, calls);
  return true;
}

void Api::remember_turn(const std::string& key, std::vector<int32_t> ids) {
  std::lock_guard<std::mutex> g(turns_mu_);
  if (!turns_.count(key)) turn_order_.push_back(key);
  turns_[key] = std::move(ids);
  while (turn_order_.size() > 256) {
    turns_.erase(turn_order_.front());
    turn_order_.pop_front();
  }
}

std::vector<int32_t> Api::prompt_ids(const Value& messages, const Value& tools, const Value& effort,
                                     int64_t* spliced) {
  RenderedPrompt rp = render_chat(messages, tools, effort);
  std::vector<int32_t> ids;
  ids.reserve(rp.text.size() / 3);
  size_t pos = 0;
  int64_t sp = 0;
  std::string_view text(rp.text);
  for (auto& span : rp.assistant_spans) {
    tok_.encode_append(text.substr(pos, span.begin - pos), ids);
    std::string key;
    const std::vector<int32_t>* hit = nullptr;
    if (o_.splice && message_key(messages.arr[span.msg_index], key)) {
      std::lock_guard<std::mutex> g(turns_mu_);
      auto it = turns_.find(key);
      if (it != turns_.end()) hit = &it->second;
      if (hit) {
        ids.insert(ids.end(), hit->begin(), hit->end());
        sp += (int64_t)hit->size();
      }
    }
    if (!hit) tok_.encode_append(text.substr(span.begin, span.end - span.begin), ids);
    pos = span.end;
  }
  tok_.encode_append(text.substr(pos), ids);
  if (spliced) *spliced = sp;
  return ids;
}

void Api::handle(const http::Request& req, http::Response& res) {
  if (req.method == "OPTIONS") {
    res.send(204, "text/plain", "");
    return;
  }
  if (req.path == "/health" || req.path == "/v1/health") {
    res.send(200, "application/json", "{\"status\":\"ok\"}");
    return;
  }
  if (req.path == "/v1/models" || req.path == "/models") {
    Value m = Value::object();
    m.set("id", o_.model_id);
    m.set("object", "model");
    m.set("created", (int64_t)0);
    m.set("owned_by", "inkling-engine");
    Value l = Value::object();
    l.set("object", "list");
    l.set("data", Value::array({m}));
    res.send(200, "application/json", pyjson::compact(l));
    return;
  }
  if (req.path == "/v1/chat/completions" || req.path == "/chat/completions") {
    if (req.method != "POST") { res.send(405, "application/json", error_json("use POST")); return; }
    chat(req, res);
    return;
  }
  if (req.path == "/tokenize" && req.method == "POST") {
    Value b;
    try { b = pyjson::parse(req.body); } catch (const std::exception& e) {
      res.send(400, "application/json", error_json(e.what())); return;
    }
    std::vector<int32_t> ids;
    try {
      if (b.find("messages")) {
        Value msgs = b["messages"];
        canonicalize_messages(msgs);
        ids = prompt_ids(msgs, b["tools"], b["reasoning_effort"]);
      } else {
        ids = tok_.encode(b.str_or("content"));
      }
    } catch (const std::exception& e) {
      res.send(400, "application/json", error_json(e.what())); return;
    }
    Value a = Value::array();
    for (int32_t id : ids) a.arr.emplace_back((int64_t)id);
    Value o = Value::object();
    o.set("tokens", std::move(a));
    res.send(200, "application/json", pyjson::compact(o));
    return;
  }
  if (req.path == "/v1/cache") {
    Value o = Value::object();
    o.set("engine", eng_.name());
    o.set("cached_tokens", (int64_t)cache_.size());
    o.set("checkpoints", (int64_t)cache_.n_ckpts());
    o.set("requests", n_requests_);
    o.set("prompt_tokens_total", total_prompt_);
    o.set("reused_tokens_total", total_reused_);
    o.set("spliced_tokens_total", total_spliced_);
    res.send(200, "application/json", pyjson::compact(o));
    return;
  }
  res.send(404, "application/json", error_json("no route " + req.path));
}

void Api::chat(const http::Request& req, http::Response& res) {
  const double t_req = now_s();
  Value body;
  try {
    body = pyjson::parse(req.body);
  } catch (const std::exception& e) {
    res.send(400, "application/json", error_json(std::string("bad JSON: ") + e.what()));
    return;
  }
  if (!body.is_object()) { res.send(400, "application/json", error_json("body must be an object")); return; }

  // ---- request fields
  Value messages = body["messages"];
  const Value& kw = body["chat_template_kwargs"];
  Value effort = o_.default_effort;
  if (const Value* e = body.find("reasoning_effort"); e && !e->is_null()) effort = *e;
  if (const Value* e = kw.find("reasoning_effort"); e && !e->is_null()) effort = *e;
  else if (const Value* t = kw.find("enable_thinking"); t && t->is_bool() && !t->b && !body.find("reasoning_effort"))
    effort = Value("none");
  const bool stream = body.bool_or("stream", false);
  const bool include_usage = body["stream_options"].bool_or("include_usage", false);
  SamplingParams sp;
  sp.temperature = (float)body.num_or("temperature", o_.default_temperature);
  sp.top_p = (float)body.num_or("top_p", o_.default_top_p);
  sp.top_k = (int)body.int_or("top_k", o_.default_top_k);
  sp.min_p = (float)body.num_or("min_p", 0.0);
  sp.seed = (uint64_t)body.int_or("seed", (int64_t)std::random_device{}());
  int64_t max_tokens = body.int_or("max_completion_tokens", body.int_or("max_tokens", o_.default_max_tokens));
  if (max_tokens <= 0) max_tokens = o_.default_max_tokens;
  const std::string model = body.str_or("model", o_.model_id);

  std::vector<int32_t> prompt;
  int64_t spliced = 0;
  try {
    canonicalize_messages(messages);
    prompt = prompt_ids(messages, body["tools"], effort, &spliced);
  } catch (const std::exception& e) {
    res.send(400, "application/json", error_json(e.what()));
    return;
  }
  const int64_t n_prompt = (int64_t)prompt.size();
  if (n_prompt + 1 > o_.max_context) {
    res.send(400, "application/json",
             error_json("prompt is " + std::to_string(n_prompt) + " tokens; the context limit is " +
                        std::to_string(o_.max_context)));
    return;
  }
  max_tokens = std::min<int64_t>(max_tokens, o_.max_context - n_prompt);

  const std::string id = new_id();
  const int64_t created = (int64_t)std::time(nullptr);
  auto chunk = [&](Value delta, const Value& finish) {
    Value ch = Value::object();
    ch.set("index", (int64_t)0);
    ch.set("delta", std::move(delta));
    ch.set("finish_reason", finish);
    Value c = Value::object();
    c.set("id", id);
    c.set("object", "chat.completion.chunk");
    c.set("created", created);
    c.set("model", model);
    c.set("choices", Value::array({ch}));
    return "data: " + pyjson::compact(c) + "\n\n";
  };
  auto tool_calls_json = [](const std::vector<ToolCallOut>& calls, size_t base, bool with_index) {
    Value a = Value::array();
    for (size_t k = 0; k < calls.size(); k++) {
      Value t = Value::object();
      if (with_index) t.set("index", (int64_t)(base + k));
      t.set("id", calls[k].id);
      t.set("type", "function");
      Value f = Value::object();
      f.set("name", calls[k].name);
      f.set("arguments", calls[k].arguments);
      t.set("function", std::move(f));
      a.arr.push_back(std::move(t));
    }
    return a;
  };

  // ---- wait for the engine
  std::unique_lock<std::mutex> lock(mu_);
  if (res.peer_closed()) return;
  if (stream) {
    if (!res.start_stream(200, "text/event-stream")) return;
    Value d = Value::object();
    d.set("role", "assistant");
    d.set("content", "");
    res.write(chunk(std::move(d), Value()));
  }

  PromptCache::Prep prep;
  OutputParser parser(tok_);
  std::vector<int32_t> emitted;
  std::string finish = "stop";
  double t_gen0 = 0, t_first = 0, t_end = 0;
  size_t n_calls_sent = 0;
  bool client_gone = false;
  try {
    double last_ka = now_s();
    std::vector<int32_t> before;
    if (o_.log) before = cache_.tokens();
    prep = cache_.prepare(prompt, [&](int64_t done, int64_t total) {
      if (stream && o_.keepalive_comments && now_s() - last_ka > 2.0) {
        last_ka = now_s();
        res.write(": prefill " + std::to_string(done) + "/" + std::to_string(total) + "\n\n");
      }
    });
    if (o_.log && prep.lcp < (int64_t)before.size() && prep.lcp + 1 < n_prompt) {
      // Where did this prompt leave the cached sequence? Text on both sides of the divergence.
      auto snip = [&](const std::vector<int32_t>& v, int64_t a, int64_t b) {
        a = std::max<int64_t>(0, a); b = std::min<int64_t>((int64_t)v.size(), b);
        std::string s = tok_.decode(std::span<const int32_t>(v.data() + a, b - a));
        for (auto& c : s) if (c == '\n') c = '|';
        return s;
      };
      std::fprintf(stderr, "[cache] diverged at %lld of cached %zu / prompt %lld\n  before: ...%s\n  cached: %s...\n  prompt: %s...\n",
                   (long long)prep.lcp, before.size(), (long long)n_prompt, snip(prompt, prep.lcp - 24, prep.lcp).c_str(),
                   snip(before, prep.lcp, prep.lcp + 24).c_str(), snip(prompt, prep.lcp, prep.lcp + 24).c_str());
    }
    t_gen0 = now_s();
    eng_.generate(sp, (int)max_tokens, stops_, [&](int32_t tok) {
      if (emitted.empty()) t_first = now_s();
      emitted.push_back(tok);
      Delta d = parser.push(tok);
      if (stream && !d.empty()) {
        Value dv = Value::object();
        if (!d.reasoning.empty()) dv.set("reasoning_content", d.reasoning);
        if (!d.content.empty()) dv.set("content", d.content);
        if (!d.tool_calls.empty()) {
          dv.set("tool_calls", tool_calls_json(d.tool_calls, n_calls_sent, true));
          n_calls_sent += d.tool_calls.size();
        }
        if (!res.write(chunk(std::move(dv), Value()))) client_gone = true;
      }
      if (!stream && (emitted.size() & 63) == 0 && res.peer_closed()) client_gone = true;
      if (stream && (emitted.size() & 15) == 0 && res.peer_closed()) client_gone = true;
      return !client_gone && !parser.done();
    });
    t_end = now_s();
    const bool turn_done = parser.done();
    if (!turn_done) {
      Delta d = parser.finish();
      if (stream && !d.empty() && !client_gone) {
        Value dv = Value::object();
        if (!d.reasoning.empty()) dv.set("reasoning_content", d.reasoning);
        if (!d.content.empty()) dv.set("content", d.content);
        if (!d.tool_calls.empty()) {
          dv.set("tool_calls", tool_calls_json(d.tool_calls, n_calls_sent, true));
          n_calls_sent += d.tool_calls.size();
        }
        res.write(chunk(std::move(dv), Value()));
      }
      if ((int64_t)emitted.size() >= max_tokens) finish = "length";
    }
    if (parser.saw_tool_call() && finish != "length") finish = "tool_calls";
    cache_.after_generate(emitted, turn_done);
    if (turn_done && o_.splice && !emitted.empty() &&
        emitted.back() == tok_.special("<|content_model_end_sampling|>")) {
      std::vector<std::pair<std::string, Value>> calls;
      for (auto& c : parser.tool_calls()) {
        Value a;
        try { a = pyjson::parse(c.arguments); } catch (...) { a = Value::object(); }
        calls.emplace_back(c.name, std::move(a));
      }
      remember_turn(turn_key(parser.reasoning(), parser.content(), calls), emitted);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[error] %s\n", e.what());
    cache_.invalidate();
    if (stream) {
      res.write("data: " + error_json(std::string("engine error: ") + e.what(), "server_error") + "\n\n");
      res.write("data: [DONE]\n\n");
    } else {
      res.send(500, "application/json", error_json(std::string("engine error: ") + e.what(), "server_error"));
    }
    return;
  }
  lock.unlock();

  n_requests_++;
  total_prompt_ += n_prompt;
  total_reused_ += prep.reused;
  total_spliced_ += spliced;
  const int64_t n_gen = (int64_t)emitted.size();
  const double gen_s = t_end - (t_first > 0 ? t_first : t_gen0);
  Value usage = Value::object();
  usage.set("prompt_tokens", n_prompt);
  usage.set("completion_tokens", n_gen);
  usage.set("total_tokens", n_prompt + n_gen);
  Value det = Value::object();
  det.set("cached_tokens", prep.reused);
  usage.set("prompt_tokens_details", std::move(det));
  Value timings = Value::object();
  timings.set("prompt_n", prep.prefilled);
  timings.set("prompt_ms", prep.prefill_s * 1000);
  timings.set("prompt_per_second", prep.prefill_s > 0 ? prep.prefilled / prep.prefill_s : 0.0);
  timings.set("predicted_n", n_gen);
  timings.set("predicted_ms", gen_s * 1000);
  timings.set("predicted_per_second", gen_s > 0 && n_gen > 1 ? (n_gen - 1) / gen_s : 0.0);
  timings.set("cache", prep.how);

  if (o_.log)
    std::fprintf(stderr,
                 "[req] prompt %lld (reused %lld via %s, spliced %lld) prefill %lld in %.3fs (%.0f tok/s) | gen %lld "
                 "in %.3fs (%.1f tok/s) | ttft %.3fs | %s%s\n",
                 (long long)n_prompt, (long long)prep.reused, prep.how.c_str(), (long long)spliced,
                 (long long)prep.prefilled, prep.prefill_s, prep.prefill_s > 0 ? prep.prefilled / prep.prefill_s : 0.0,
                 (long long)n_gen, gen_s, gen_s > 0 && n_gen > 1 ? (n_gen - 1) / gen_s : 0.0,
                 (t_first > 0 ? t_first : t_end) - t_req, finish.c_str(), client_gone ? " (client gone)" : "");

  if (client_gone) return;
  if (stream) {
    res.write(chunk(Value::object(), Value(finish)));
    if (include_usage) {
      Value c = Value::object();
      c.set("id", id);
      c.set("object", "chat.completion.chunk");
      c.set("created", created);
      c.set("model", model);
      c.set("choices", Value::array());
      c.set("usage", usage);
      c.set("timings", timings);
      res.write("data: " + pyjson::compact(c) + "\n\n");
    }
    res.write("data: [DONE]\n\n");
    return;
  }
  Value msg = Value::object();
  msg.set("role", "assistant");
  msg.set("content", parser.content().empty() && parser.saw_tool_call() ? Value() : Value(parser.content()));
  if (!parser.reasoning().empty()) msg.set("reasoning_content", parser.reasoning());
  if (parser.saw_tool_call()) msg.set("tool_calls", tool_calls_json(parser.tool_calls(), 0, false));
  Value ch = Value::object();
  ch.set("index", (int64_t)0);
  ch.set("message", std::move(msg));
  ch.set("finish_reason", finish);
  Value out = Value::object();
  out.set("id", id);
  out.set("object", "chat.completion");
  out.set("created", created);
  out.set("model", model);
  out.set("choices", Value::array({ch}));
  out.set("usage", usage);
  out.set("timings", timings);
  res.send(200, "application/json", pyjson::compact(out));
}

}  // namespace ink
