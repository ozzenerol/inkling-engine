// inkling-server: OpenAI-compatible server for Inkling 2 Mini. Owner: C.
//   inkling-server [--model DIR] [--host 127.0.0.1] [--port 8091] [--stub] [--ctx 131072] [--max-tokens 16384]
//                  [--temp 1.0] [--effort none|low|...|0.5] [--no-mtp] [--draft-block 4] [--prefill-chunk 2048]
//                  [--no-splice] [--ckpts 4]
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "api.h"

using namespace ink;

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  std::string model = std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
                      "/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit";
  std::string host = "127.0.0.1";
  int port = 8091;
  bool stub = false;
  ApiOptions ao;
  EngineOptions eo;
  StubOptions so;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--host") host = next();
    else if (a == "--port") port = std::stoi(next());
    else if (a == "--stub") stub = true;
    else if (a == "--stub-decode-ms") so.decode_ms_per_token = std::stod(next());
    else if (a == "--ctx") ao.max_context = eo.max_context = std::stoll(next());
    else if (a == "--max-tokens") ao.default_max_tokens = std::stoi(next());
    else if (a == "--temp") ao.default_temperature = std::stof(next());
    else if (a == "--effort") {
      std::string e = next();
      char* end;
      double d = std::strtod(e.c_str(), &end);
      ao.default_effort = *end == 0 && !e.empty() ? json::Value(d) : json::Value(e);
    }
    else if (a == "--no-mtp") eo.use_mtp = false;
    else if (a == "--draft-block") eo.draft_block = std::stoi(next());
    else if (a == "--prefill-chunk") eo.prefill_chunk = std::stoi(next());
    else if (a == "--no-splice") ao.splice = false;
    else if (a == "--quiet") ao.log = false;
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
  }
  eo.model_dir = model;
  ao.model_id = model;

  std::unique_ptr<Tokenizer> tok;
  try {
    tok = Tokenizer::load(model);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tokenizer: %s\n", e.what());
    return 1;
  }
  std::unique_ptr<EngineIface> eng;
  if (!stub) {
    std::fprintf(stderr, "loading model from %s ...\n", model.c_str());
    try {
      eng = make_real_engine(eo);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "engine: %s\n", e.what());
      return 1;
    }
    if (!eng) { std::fprintf(stderr, "built without the engine; use --stub\n"); return 1; }
  } else {
    eng = make_stub_engine(*tok, so);
  }
  Api api(*tok, *eng, ao);
  std::fprintf(stderr, "inkling-server (%s engine) on http://%s:%d/v1\n", eng->name().c_str(), host.c_str(), port);
  try {
    http::serve(host, port, [&](const http::Request& rq, http::Response& rs) { api.handle(rq, rs); });
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
