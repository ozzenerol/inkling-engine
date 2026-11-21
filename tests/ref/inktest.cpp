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
#include <algorithm>
#include <cmath>
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
  std::string isolate;
  int debug_visit = -1;
  bool rollback_test = false;  // prefix of ref hidden files: PREFIX{v}.f32
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
    else if (a == "--isolate") isolate = next();
    else if (a == "--debug-visit") debug_visit = std::stoi(next());
    else if (a == "--rollback-test") rollback_test = true;
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
    if (rollback_test) {
      // Speculative rollback: [prefix][4 rows with history] -> keep 2 -> [3 more] must equal the straight run
      // [prefix][2][3] (bit-identical on Metal where rows are independent; CPU differs by BLAS summation order).
      const int64_t n0 = std::min<int64_t>(int64_t(ids.size()) - 7, 12);
      if (n0 < 1) throw std::runtime_error("rollback test needs at least 8 ids");
      std::vector<std::vector<std::vector<float>>> snap;
      auto run = [&](bool spec) {
        m.reset();
        m.forward(ids.data(), n0, false);
        if (spec) {
          int32_t four[4] = {ids[size_t(n0)], ids[size_t(n0 + 1)], 7, 9};  // rows 2, 3 are rejected drafts
          m.forward(four, 4, true, true);
          m.rollback(2);
        } else {
          m.forward(ids.data() + n0, 2, false);
        }
        be->sync();
        snap.emplace_back();
        for (auto& vs : m.visits())
          for (int i = 0; i < 4; ++i) {
            std::vector<float> c(size_t(vs.conv[i].numel()));
            be->read(vs.conv[i], c.data(), c.size() * 4);
            snap.back().push_back(c);
          }
        m.forward(ids.data() + n0 + 2, 3, true);
        return m.logits();
      };
      auto a1 = run(true), a2 = run(false);
      for (size_t k = 0; k < snap[0].size(); ++k) {
        double d = 0;
        for (size_t i = 0; i < snap[0][k].size(); ++i) d = std::max(d, double(std::fabs(snap[0][k][i] - snap[1][k][i])));
        if (d > 1e-3) { std::printf("  conv state differs: visit %zu conv %zu max diff %.4g\n", k / 4, k % 4, d); if (k > 12) break; }
      }
      double maxd = 0;
      size_t same = 0;
      int argdiff = 0;
      for (size_t i = 0; i < a1.size(); ++i) { maxd = std::max(maxd, double(std::fabs(a1[i] - a2[i]))); same += a1[i] == a2[i]; }
      for (int r = 0; r < 3; ++r) {
        auto b1 = a1.begin() + r * V, b2 = a2.begin() + r * V;
        argdiff += (std::max_element(b1, b1 + V) - b1) != (std::max_element(b2, b2 + V) - b2);
      }
      std::printf("rollback test (%s): max |dlogit| %.4g, identical %.2f%%, argmax differences %d/3, len %lld\n", be->name(),
                  maxd, 100.0 * same / a1.size(), argdiff, (long long)m.len());
      return (argdiff == 0 && maxd < 0.5 && m.len() == n0 + 5) ? 0 : 1;
    }
    if (!isolate.empty()) {
      // Per-visit isolation: run each visit on the reference input/blocks/bank, compare with the next ref input.
      const auto& cfg = m.config();
      const int64_t T = int64_t(ids.size());
      auto load_ref = [&](int v) {
        std::ifstream f(isolate + std::to_string(v) + ".f32", std::ios::binary);
        std::vector<float> h(size_t(T * D));
        f.read(reinterpret_cast<char*>(h.data()), h.size() * 4);
        return h;
      };
      auto to_dev = [&](const std::vector<float>& h) {
        Tensor t = be->alloc(DType::BF16, {T, D}, "iso");
        std::vector<uint16_t> b(h.size());
        for (size_t i = 0; i < h.size(); ++i) b[i] = f32_to_bf16(h[i]);
        be->write(t, b.data(), b.size() * 2);
        return t;
      };
      if (debug_visit >= 0)
        m.on_debug = [&](int v, const char* name, const Tensor& t) {
          if (v != debug_visit) return;
          be->sync();
          std::vector<float> f(size_t(t.numel()));
          if (t.dtype == DType::F32) be->read(t, f.data(), f.size() * 4);
          else {
            std::vector<uint16_t> h(f.size());
            be->read(t, h.data(), h.size() * 2);
            for (size_t i = 0; i < h.size(); ++i) f[i] = bf16_to_f32(h[i]);
          }
          std::ofstream o(out + "/dbg_" + name + ".f32", std::ios::binary);
          o.write(reinterpret_cast<const char*>(f.data()), f.size() * 4);
        };
      std::vector<Tensor> ref(size_t(cfg.num_visits()));
      for (int v = 0; v < cfg.num_visits(); ++v) ref[size_t(v)] = to_dev(load_ref(v));
      std::vector<int> blocks;  // visit indices whose ref input is a block
      int prelude_n = -1, prev12 = -1;
      Tensor xo = be->alloc(DType::BF16, {T, D}, "iso.out"), last = be->alloc(DType::BF16, {T, D}, "iso.last");
      for (int v = 0; v < cfg.num_visits(); ++v) {
        const int L = cfg.visits[size_t(v)];
        std::vector<const Tensor*> bank;
        if (L == cfg.core_start) {
          if (prelude_n < 0) prelude_n = int(blocks.size());
          else { blocks.resize(size_t(prelude_n)); }
        }
        if (v >= cfg.core_start + (cfg.core_end - cfg.core_start) && L >= cfg.core_start && L < cfg.core_end) {
          bank = {&ref[size_t(prev12)], &last};
        }
        std::vector<const Tensor*> bl;
        for (int b : blocks) bl.push_back(&ref[size_t(b)]);
        if (v == debug_visit && !bank.empty()) {
          be->sync();
          std::vector<uint16_t> h(size_t(T * D));
          be->read(last, h.data(), h.size() * 2);
          std::vector<float> f(h.size());
          for (size_t i = 0; i < h.size(); ++i) f[i] = bf16_to_f32(h[i]);
          std::ofstream o(out + "/bank1.f32", std::ios::binary);
          o.write(reinterpret_cast<const char*>(f.data()), f.size() * 4);
        }
        m.debug_visit(v, ref[size_t(v)], bl, bank, xo, T);
        if (L % cfg.block == 0) blocks.push_back(v);
        if (L == 12) prev12 = v;
        if (L == cfg.core_end - 1) be->copy(xo, last);
        // compare with next ref input (skip pass restarts where the next input is the prelude)
        const bool restart = v + 1 < cfg.num_visits() && cfg.visits[size_t(v + 1)] == cfg.core_start && v + 1 != cfg.core_start;
        if (v + 1 < cfg.num_visits() && !restart) {
          std::vector<uint16_t> o(size_t(T * D));
          be->sync();
          be->read(xo, o.data(), o.size() * 2);
          auto r = load_ref(v + 1);
          double num = 0, den = 0, mx = 0;
          size_t neq = 0;
          for (size_t i = 0; i < o.size(); ++i) {
            double d = double(bf16_to_f32(o[i])) - r[i];
            num += d * d; den += double(r[i]) * r[i]; mx = std::max(mx, std::fabs(d));
            neq += f32_to_bf16(r[i]) != o[i];
          }
          std::printf("visit %2d (layer %2d %s): rel err %.3e  max abs %.3e  bf16 mismatch %.2f%%\n", v, L,
                      cfg.is_local[size_t(L)] ? "local " : "global", std::sqrt(num / den), mx, 100.0 * neq / o.size());
        }
      }
      return 0;
    }
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
