// Owner: A. Teacher-forced reference run: feeds the reference prompt (chunked prefill) and then the reference
// greedy tokens one by one, writing our logits for every step (and optionally the hidden state entering every
// visit at prefill). tests/ref/compare.py compares them with the MLX dump.
//
//   inktest --backend cpu|metal --model DIR --ids p0.ids --gen p0.gen --steps N --out OUTDIR [--hidden]
//           [--chunk 2048] [--ctx 8192]
#include <execinfo.h>
#include <csignal>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "core/bf16.h"
#include "model/backend.h"
#include "model/model.h"

using namespace ink;

static std::vector<int32_t> read_i32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  std::vector<int32_t> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), v.size() * 4);
  return v;
}

static void on_segv(int sig) {
  void* fr[64];
  int n = backtrace(fr, 64);
  backtrace_symbols_fd(fr, n, 2);
  _exit(128 + sig);
}

int main(int argc, char** argv) {
  std::signal(SIGSEGV, on_segv);
  std::signal(SIGBUS, on_segv);
  std::string backend = "cpu", model = std::string(getenv("HOME")) + "/llm/mlx-models/Inkling-2-Mini-MLX-4-8bit";
  std::string ids_path, gen_path, out;
  int steps = 8;
  int64_t chunk = 2048, ctx = 8192;
  bool hidden = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&] { return std::string(argv[++i]); };
    if (a == "--backend") backend = next();
    else if (a == "--model") model = next();
    else if (a == "--ids") ids_path = next();
    else if (a == "--gen") gen_path = next();
    else if (a == "--steps") steps = std::stoi(next());
    else if (a == "--out") out = next();
    else if (a == "--hidden") hidden = true;
    else if (a == "--chunk") chunk = std::stoll(next());
    else if (a == "--ctx") ctx = std::stoll(next());
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  try {
    auto ids = read_i32(ids_path);
    auto gen = gen_path.empty() ? std::vector<int32_t>{} : read_i32(gen_path);
    auto be = backend == "cpu" ? make_cpu_backend() : make_metal_backend();
    auto t0 = std::chrono::steady_clock::now();
    Model m(*be, model, ctx, chunk);
    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[inktest] backend %s, load %.1f s\n", be->name(), std::chrono::duration<double>(t1 - t0).count());
    const int64_t D = m.config().hidden, V = m.config().vocab_unpadded;
    std::vector<std::vector<float>> hid(static_cast<size_t>(m.config().num_visits() + 1));
    int64_t pos_base = 0;
    if (hidden) {
      m.on_visit = [&](int v, const Tensor& x) {
        be->sync();
        const int64_t T = x.numel() / D;
        std::vector<float> f(static_cast<size_t>(T * D));
        if (x.dtype == DType::F32) {
          be->read(x, f.data(), f.size() * 4);
        } else {
          std::vector<uint16_t> h(f.size());
          be->read(x, h.data(), h.size() * 2);
          for (size_t i = 0; i < h.size(); ++i) f[i] = bf16_to_f32(h[i]);
        }
        auto& dst = hid[size_t(v)];
        dst.insert(dst.end(), f.begin(), f.end());
        (void)pos_base;
      };
    }
    std::ofstream lf(out + "/logits.f32", std::ios::binary);
    std::vector<int32_t> ours;
    auto t2 = std::chrono::steady_clock::now();
    for (int64_t p = 0; p < int64_t(ids.size()); p += chunk) {
      int64_t T = std::min<int64_t>(chunk, int64_t(ids.size()) - p);
      m.forward(ids.data() + p, T, false);
    }
    m.on_visit = nullptr;
    auto t3 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[inktest] prefill %zu tokens %.2f s\n", ids.size(), std::chrono::duration<double>(t3 - t2).count());
    for (int s = 0; s < steps; ++s) {
      const auto& lg = m.logits();
      lf.write(reinterpret_cast<const char*>(lg.data()), V * 4);
      int32_t best = 0;
      for (int64_t i = 1; i < V; ++i) if (lg[i] > lg[best]) best = int32_t(i);
      ours.push_back(best);
      if (s + 1 == steps) break;
      int32_t feed = s < int(gen.size()) ? gen[s] : best;  // teacher forcing with the reference token
      m.forward(&feed, 1, false);
    }
    auto t4 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[inktest] %d decode steps %.2f s (%.2f tok/s)\n", steps - 1,
                 std::chrono::duration<double>(t4 - t3).count(),
                 (steps - 1) / std::chrono::duration<double>(t4 - t3).count());
    std::ofstream gf(out + "/argmax.i32", std::ios::binary);
    gf.write(reinterpret_cast<const char*>(ours.data()), ours.size() * 4);
    if (hidden)
      for (size_t v = 0; v < hid.size(); ++v) {
        std::ofstream hf(out + "/hidden_" + std::to_string(v) + ".f32", std::ios::binary);
        hf.write(reinterpret_cast<const char*>(hid[v].data()), hid[v].size() * 4);
      }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
