// DSpark drafter on Metal. Owner: B. See dspark.h. Generic ops go through the Backend (MetalBackend), the
// drafter-specific kernels (dspark.metal, split attention) are dispatched on the same command buffer.
#include "gpu/dspark.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

#include "core/bf16.h"
#include "core/safetensors.h"
#include "gpu/context.h"
#include "model/backend.h"

namespace ink {

bool metal_dequant_affine(Backend& be, const Tensor& codes, Tensor& out);  // metal_backend.mm

const int DSpark::kCaptureVisits[DSpark::kCaptures] = {5, 23, 41, 48};

namespace {
using gpu::Buf;
using gpu::Context;

Buf bufof(const Tensor& t) {
  Buf b;
  b.mtl = t.buf;
  b.host = t.host;
  return b;
}
uint32_t cdiv(uint64_t a, uint64_t b) { return uint32_t((a + b - 1) / b); }

struct DMixParams {
  uint32_t L, D;
  float alpha[4];
  float eps;
  uint32_t row_stride;
};
struct RopeParams {
  uint32_t T, H, pos0;
  float log2_base;
};
struct MarkovParams {
  uint32_t rank, prev_index;
};
struct FoldParams {
  uint32_t V, R;
  float cap;
  uint32_t write_z;
};
struct SampleParams {
  uint32_t V, nparts, out_index, step;
  float inv_temp, u;
};
struct ArgmaxParams {
  uint32_t n, out_index;
};
struct GaParams {  // kernels/multirow_ga.metal (D)
  uint32_t T, H, KVH, cap, len, rel_extent, sr, sk, nblk, nsplit;
  float thr;
  uint32_t use_sink, use_gate, causal;
};
struct AttnDecParams {
  uint32_t T, H, KVH, cap, len, ring, window, rel_extent, split_keys, nsplit, causal, first, use_sink, use_gate;
};

Tensor view(const Tensor& t, std::initializer_list<int64_t> shape, int64_t first_row = 0) {
  Tensor v = t;
  v.set_shape(shape);
  v.offset = t.offset + size_t(first_row) * v.nbytes() / size_t(std::max<int64_t>(1, *shape.begin()));
  return v;
}
}  // namespace

struct DSpark::Impl {
  Backend& be;
  Context& g = Context::get();
  int64_t D = 2304, H = 24, KVH = 8, HD = 128, I = 9216, R = 256, V = 201024, Vu = 200058;
  int64_t window = 1024, slack = 512, max_ctx, max_rows;
  float eps = 1e-6f, theta = 1e6f, cap = 30.0f;
  struct Layer {
    bool local;
    Tensor in_ln, post_ln, dflare, q, k, v, kc, vc, o, qn, kn, kcn, gate, up, down;
    KvView kv;
  };
  std::vector<Layer> L;
  Tensor norm, mask, fold_w, fold_b, ws, wc, rnn_b, w1, mk_dflare;
  float alpha[4][4];
  const Tensor &embed, &embed_norm, &head;
  // workspace
  Tensor mixed, kx, vx, tokens, e0, e1, x, xin, q, kb, vb, at, o, summed, res, post, gt, ut, a, h, base, frnn, cat,
      ctxm, state, zero_state, latent, part, last_feat, apart, zbuf, stats, qbuf, ga_s, ga_bmax, ga_m, ga_part;
  // D: direct global attention (multirow_ga.metal) for the drafter's global layer: on unless INK_GA=0 or
  // INK_GA_DRAFT=0
  bool ga = !(getenv("INK_GA") && getenv("INK_GA")[0] == '0') && !(getenv("INK_GA_DRAFT") && getenv("INK_GA_DRAFT")[0] == '0');
  float last_temp = 0;

  Impl(Backend& b, const Tensor& em, const Tensor& en, const Tensor& hd) : be(b), embed(em), embed_norm(en), head(hd) {}

  void bind(uint32_t i, const Tensor& t) { g.set_buffer(i, bufof(t), t.offset); }
  void pipe(const char* n) { g.set_pipeline(g.pipeline(n)); }

  void mix(const Tensor& feat, int64_t rows, int which, const Tensor& gamma, Tensor& out) {
    DMixParams p{uint32_t(rows), uint32_t(D), {alpha[which][0], alpha[which][1], alpha[which][2], alpha[which][3]},
                 eps, uint32_t(4 * D)};
    pipe("dspark_mix");
    bind(0, feat);
    bind(1, gamma);
    bind(2, out);
    g.set_value(3, p);
    g.dispatch({uint32_t(rows), 1, 1}, {256, 1, 1});
  }
  void rope(Tensor& t, int64_t T, int64_t heads, int64_t pos0) {
    RopeParams p{uint32_t(T), uint32_t(heads), uint32_t(pos0), std::log2(theta)};
    pipe("dspark_rope");
    bind(0, t);
    g.set_value(1, p);
    g.dispatch({1, uint32_t(heads), uint32_t(T)}, {64, 1, 1});
  }
  // Bidirectional attention of T block rows over the layer's context KV plus the block's own keys.
  void attend(Layer& l, const Tensor& qt, const Tensor& kn, const Tensor& vn, int64_t T, Tensor& out) {
    uint32_t len = uint32_t(l.kv.len);
    if (ga && !l.local && !l.kv.ring && T <= 16) {
      // D: the global layer through the direct two-pass kernels (K and V read once from device memory, no staging;
      // at 126k the split kernel made a draft round cost 17 ms). Drafts only change speed, so no identity rule here.
      const uint32_t Tu = uint32_t(T), nkeys = len + Tu, sr = 3 * Tu;
      const uint32_t tok = Tu == 1 ? 1 : Tu == 2 ? 2 : Tu <= 4 ? 4 : Tu <= 8 ? 8 : 16;
      GaParams p{Tu, uint32_t(H), uint32_t(KVH), uint32_t(l.kv.cap), len, 0u, sr, cdiv(nkeys, 64) * 64,
                 cdiv(nkeys, 64), cdiv(nkeys, 512), 1e30f, 0u, 0u, 0u};
      const int64_t ska = int64_t(cdiv(nkeys, 16384)) * 16384, sra = 48;
      if (ga_s.numel() < int64_t(KVH) * sr * ska) {
        be.sync();
        for (Tensor* t : {&ga_s, &ga_bmax, &ga_m, &ga_part})
          if (t->valid()) be.free(*t);
        ga_s = be.alloc(DType::F32, {KVH * sra, ska}, "dspark.ga_s");
        ga_bmax = be.alloc(DType::F32, {KVH * sra, ska / 64}, "dspark.ga_bmax");
        ga_m = be.alloc(DType::F32, {KVH * sra}, "dspark.ga_m");
        ga_part = be.alloc(DType::F32, {KVH * (ska / 512) * sra, 132}, "dspark.ga_part");
      }
      pipe("ga_scores");
      bind(0, qt);
      bind(1, kn);
      bind(2, qt);  // r, rel_proj unused (rel_extent 0)
      bind(3, qt);
      bind(4, l.kv.k);
      bind(5, ga_s);
      bind(6, ga_bmax);
      g.set_value(7, p);
      g.set_value(8, tok);
      g.dispatch({p.nblk, uint32_t(KVH), cdiv(Tu, tok)}, {32 * tok, 1, 1});
      pipe("ga_rowmax");
      bind(0, ga_bmax);
      bind(1, qt);  // sink unused
      bind(2, ga_m);
      g.set_value(3, p);
      g.dispatch({sr, uint32_t(KVH), 1}, {32, 1, 1});
      pipe("ga_pv");
      bind(0, ga_s);
      bind(1, ga_bmax);
      bind(2, ga_m);
      bind(3, vn);
      bind(4, l.kv.v);
      bind(5, ga_part);
      g.set_value(6, p);
      g.set_value(7, tok);
      g.dispatch({p.nsplit, uint32_t(KVH), cdiv(Tu, tok)}, {32 * tok, 1, 1});
      pipe("ga_combine");
      bind(0, ga_part);
      bind(1, ga_m);
      bind(2, qt);
      bind(3, out);
      bind(4, out);
      g.set_value(5, p);
      g.dispatch({Tu, uint32_t(H), 1}, {128, 1, 1});
      return;
    }
    uint32_t first = (l.local && len > window) ? uint32_t(len - window) : 0;
    uint32_t nkeys = len + uint32_t(T) - first;
    uint32_t split_keys = 512;
    uint32_t nsplit = cdiv(len + uint32_t(T), split_keys) - first / split_keys;
    (void)nkeys;
    AttnDecParams p{uint32_t(T), uint32_t(H), uint32_t(KVH), uint32_t(l.kv.cap), len, l.kv.ring ? 1u : 0u, 0u, 0u,
                    split_keys, nsplit, 0u, first, 0u, 0u};
    if (apart.numel() < int64_t(KVH * nsplit * 24 * 132)) {
      be.sync();
      if (apart.valid()) be.free(apart);
      apart = be.alloc(DType::F32, {int64_t(KVH * nsplit * 24), 132}, "dspark.attnpart");
    }
    pipe("attn_decode_split_f3");
    bind(0, qt);
    bind(1, kn);
    bind(2, vn);
    bind(3, qt);  // r unused (rel_extent 0)
    bind(4, qt);
    bind(5, l.kv.k);
    bind(6, l.kv.v);
    bind(7, apart);
    g.set_value(8, p);
    g.dispatch({uint32_t(KVH), nsplit, 1}, {64, 1, 1});
    pipe("attn_decode_combine");
    bind(0, apart);
    bind(1, qt);  // sink unused
    bind(2, out);
    g.set_value(3, p);
    bind(4, out);
    g.dispatch({uint32_t(T), uint32_t(H), 1}, {128, 1, 1});
  }
};

DSpark::DSpark(Backend& be, const std::string& dir, const Tensor& embed, const Tensor& embed_norm,
               const Tensor& lm_head, int64_t max_ctx, int64_t max_rows)
    : impl_(std::make_unique<Impl>(be, embed, embed_norm, lm_head)) {
  auto& m = *impl_;
  m.max_ctx = max_ctx;
  m.max_rows = max_rows;
  // mtp.safetensors next to the model; the 8-bit build ships without it, so fall back to INK_MTP_DIR or the
  // sibling 4-8bit build (the drafter is bf16 and identical across builds).
  std::string mtp = dir + "/mtp.safetensors";
  auto exists = [](const std::string& f) { return access(f.c_str(), R_OK) == 0; };
  if (!exists(mtp)) {
    std::vector<std::string> alts;
    if (const char* e = getenv("INK_MTP_DIR")) alts.push_back(std::string(e) + "/mtp.safetensors");
    std::string parent = dir.substr(0, dir.find_last_of('/'));
    alts.push_back(parent + "/Inkling-2-Mini-MLX-4-8bit/mtp.safetensors");
    for (auto& a : alts)
      if (exists(a)) {
        mtp = a;
        break;
      }
    if (!exists(mtp)) throw std::runtime_error("no mtp.safetensors (set INK_MTP_DIR)");
    fprintf(stderr, "dspark: using %s\n", mtp.c_str());
  }
  SafeTensors st;
  st.add_file(mtp);
  auto W = [&](const std::string& n) { return be.load_weight(st.at(n)); };
  const int nl = 3;
  const bool local[3] = {true, false, true};  // mtp_config.local_layer_ids = [0, 2]
  for (int i = 0; i < nl; i++) {
    std::string p = "layers." + std::to_string(i) + ".";
    Impl::Layer l;
    l.local = local[i];
    l.in_ln = W(p + "input_layernorm.weight");
    l.post_ln = W(p + "post_attention_layernorm.weight");
    l.dflare = W(p + "dflare_norm.weight");
    l.q = W(p + "self_attn.q_proj.weight");
    l.k = W(p + "self_attn.k_proj.weight");
    l.v = W(p + "self_attn.v_proj.weight");
    l.kc = W(p + "self_attn.k_proj_ctx.weight");
    l.vc = W(p + "self_attn.v_proj_ctx.weight");
    l.o = W(p + "self_attn.o_proj.weight");
    l.qn = W(p + "self_attn.q_norm.weight");
    l.kn = W(p + "self_attn.k_norm.weight");
    l.kcn = W(p + "self_attn.k_norm_ctx.weight");
    l.gate = W(p + "mlp.gate_proj.weight");
    l.up = W(p + "mlp.up_proj.weight");
    l.down = W(p + "mlp.down_proj.weight");
    int64_t cap = l.local ? m.window + m.slack : max_ctx;
    l.kv.k = be.alloc(DType::BF16, {m.KVH, cap, m.HD}, "dspark.k");
    l.kv.v = be.alloc(DType::BF16, {m.KVH, cap, m.HD}, "dspark.v");
    l.kv.cap = cap;
    l.kv.len = 0;
    l.kv.ring = l.local;
    m.L.push_back(l);
  }
  m.norm = W("norm.weight");
  m.mask = W("mask_embed");
  m.mk_dflare = W("markov_head.dflare_norm.weight");
  m.w1 = W("markov_head.markov_w1.weight");
  // fuse weights: alpha = bf16(softmax(fuse_weight, axis=-1))
  {
    const auto& e = st.at("fuse_weight");
    uint16_t fw[16];
    read_entry(e, fw);
    for (int r = 0; r < 4; r++) {
      float v[4], mx = -INFINITY, z = 0;
      for (int c = 0; c < 4; c++) mx = std::max(mx, v[c] = bf16_to_f32(fw[r * 4 + c]));
      for (int c = 0; c < 4; c++) z += (v[c] = std::exp(v[c] - mx));
      for (int c = 0; c < 4; c++) m.alpha[r][c] = round_bf16(v[c] / z);
    }
  }
  // rnn_proj [768, 5120] -> ws = columns 0..511, wc = columns 512..5119; bias
  {
    const auto& e = st.at("markov_head.rnn_proj.weight");
    std::vector<uint16_t> w(768 * 5120), a(768 * 512), b(768 * 4608);
    read_entry(e, w.data());
    for (int r = 0; r < 768; r++) {
      memcpy(&a[r * 512], &w[r * 5120], 512 * 2);
      memcpy(&b[r * 4608], &w[r * 5120 + 512], 4608 * 2);
    }
    m.ws = be.alloc(DType::BF16, {768, 512}, "dspark.ws");
    m.wc = be.alloc(DType::BF16, {768, 4608}, "dspark.wc");
    be.write(m.ws, a.data(), a.size() * 2);
    be.write(m.wc, b.data(), b.size() * 2);
    m.rnn_b = W("markov_head.rnn_proj.bias");
  }
  // fold: fold_w = bf16(head @ hidden_out.weight) [V, 256], fold_b = bf16(head @ hidden_out.bias) [V]
  {
    const auto& e = st.at("markov_head.hidden_out.weight");  // [2304, 256]
    std::vector<uint16_t> w(2304 * 256), t(256 * 2304);
    read_entry(e, w.data());
    for (int r = 0; r < 2304; r++)
      for (int c = 0; c < 256; c++) t[c * 2304 + r] = w[r * 256 + c];
    Tensor wt = be.alloc(DType::BF16, {256, 2304}, "dspark.tmp");
    be.write(wt, t.data(), t.size() * 2);
    Tensor bias = W("markov_head.hidden_out.bias");
    Tensor b2 = bias;
    b2.set_shape({1, 2304});
    m.fold_w = be.alloc(DType::BF16, {m.V, 256}, "dspark.fold_w");
    m.fold_b = be.alloc(DType::BF16, {m.V, 1}, "dspark.fold_b");
    Tensor head_bf = lm_head, tmp_head;
    if (lm_head.dtype == DType::U32) {  // 8-bit build: a temporary bf16 head for the one-time fold
      tmp_head = be.alloc(DType::BF16, {lm_head.dim(0), lm_head.dim(1) * 4}, "dspark.head_tmp");
      if (!metal_dequant_affine(be, lm_head, tmp_head)) throw std::runtime_error("dspark: q8 head needs Metal");
      head_bf = tmp_head;
    }
    be.linear(head_bf, wt, m.fold_w);
    be.linear(head_bf, b2, m.fold_b);
    be.sync();
    be.free(wt);
    if (tmp_head.valid()) be.free(tmp_head);
  }
  // workspace
  const int64_t B = 8;
  int64_t rows = std::max<int64_t>(max_rows, B);
  m.mixed = be.alloc(DType::BF16, {rows, m.D}, "dspark.mixed");
  m.kx = be.alloc(DType::BF16, {rows, m.KVH * m.HD}, "dspark.kx");
  m.vx = be.alloc(DType::BF16, {rows, m.KVH * m.HD}, "dspark.vx");
  m.tokens = be.alloc(DType::I32, {B}, "dspark.tokens");
  m.e0 = be.alloc(DType::BF16, {1, m.D}, "dspark.e0");
  m.e1 = be.alloc(DType::BF16, {1, m.D}, "dspark.e1");
  m.x = be.alloc(DType::BF16, {B, m.D}, "dspark.x");
  m.xin = be.alloc(DType::BF16, {B, m.D}, "dspark.xin");
  m.q = be.alloc(DType::BF16, {B, m.H * m.HD}, "dspark.q");
  m.kb = be.alloc(DType::BF16, {B, m.KVH * m.HD}, "dspark.kb");
  m.vb = be.alloc(DType::BF16, {B, m.KVH * m.HD}, "dspark.vb");
  m.at = be.alloc(DType::BF16, {B, m.H * m.HD}, "dspark.at");
  m.o = be.alloc(DType::BF16, {B, m.D}, "dspark.o");
  m.summed = be.alloc(DType::F32, {B, m.D}, "dspark.summed");
  m.res = be.alloc(DType::BF16, {B, m.D}, "dspark.res");
  m.post = be.alloc(DType::BF16, {B, m.D}, "dspark.post");
  m.gt = be.alloc(DType::BF16, {B, m.I}, "dspark.g");
  m.ut = be.alloc(DType::BF16, {B, m.I}, "dspark.u");
  m.a = be.alloc(DType::BF16, {B, m.I}, "dspark.a");
  m.h = be.alloc(DType::BF16, {B, m.D}, "dspark.h");
  m.base = be.alloc(DType::BF16, {B, m.V}, "dspark.base");
  m.frnn = be.alloc(DType::BF16, {1, m.D}, "dspark.frnn");
  m.cat = be.alloc(DType::BF16, {B, 2 * m.D}, "dspark.cat");
  m.ctxm = be.alloc(DType::F32, {B, 3 * m.R}, "dspark.ctxm");
  m.state = be.alloc(DType::BF16, {m.R}, "dspark.state");
  m.zero_state = be.alloc(DType::BF16, {m.R}, "dspark.zero");  // alloc is zero-filled
  m.latent = be.alloc(DType::BF16, {m.R}, "dspark.latent");
  m.part = be.alloc(DType::F32, {int64_t(cdiv(m.V, 32)), 2}, "dspark.part");
  m.last_feat = be.alloc(DType::BF16, {1, 4 * m.D}, "dspark.last");
  m.zbuf = be.alloc(DType::F32, {B, m.Vu}, "dspark.z");
  m.stats = be.alloc(DType::F32, {B, 4}, "dspark.stats");
  cap_ = be.alloc(DType::BF16, {rows, 4, m.D}, "dspark.capture");
}

DSpark::~DSpark() = default;

void DSpark::capture_rows(const Tensor& x, int slot) {
  auto& m = *impl_;
  struct CapParams {
    uint32_t T, D, slot, dt_src;
  } p{uint32_t(x.numel() / m.D), uint32_t(m.D), uint32_t(slot), x.dtype == DType::F32 ? 0u : 1u};
  if (int64_t(p.T) > m.max_rows) throw std::runtime_error("dspark: capture rows > max_rows");
  m.pipe("dspark_capture");
  m.bind(0, x);
  m.bind(1, cap_);
  m.g.set_value(2, p);
  m.g.dispatch({cdiv(m.D, 256), p.T, 1}, {256, 1, 1});
}

void DSpark::truncate(int64_t n) {
  if (n == ctx_len_) return;
  ctx_len_ = n;  // n > ctx_len_ leaves a gap of stale keys (restore to a later checkpoint): draft quality only
  for (auto& l : impl_->L) l.kv.len = n;
}

void DSpark::ingest(int64_t n) {
  if (n <= 0) return;
  auto& m = *impl_;
  for (auto& l : m.L) {
    Tensor feat = view(cap_, {n, 4, m.D});
    Tensor mx = view(m.mixed, {n, m.D});
    m.mix(feat, n, int(&l - &m.L[0]), l.dflare, mx);
    // ring layers only need the last `window` rows; rows written to the same slot in one dispatch would race
    int64_t skip = l.local ? std::max<int64_t>(0, n - m.window) : 0;
    int64_t k = n - skip;
    Tensor mxk = mx;
    mxk.set_shape({k, m.D});
    mxk.offset += size_t(skip) * m.D * 2;
    Tensor kx = view(m.kx, {k, m.KVH * m.HD}), vx = view(m.vx, {k, m.KVH * m.HD});
    m.be.linear(mxk, l.kc, kx);
    m.be.linear(mxk, l.vc, vx);
    Tensor k3 = view(m.kx, {k, m.KVH, m.HD});
    m.be.rms_norm(k3, &l.kcn, m.eps, m.HD, k3);
    m.rope(k3, k, m.KVH, ctx_len_ + skip);
    Tensor v3 = view(m.vx, {k, m.KVH, m.HD});
    l.kv.len = ctx_len_ + skip;
    m.be.kv_append(k3, v3, l.kv);
  }
  // last row's features for the Markov head
  Tensor lastrow = cap_;
  lastrow.set_shape({1, 4 * m.D});
  lastrow.offset += size_t(n - 1) * 4 * m.D * 2;
  m.be.copy(lastrow, m.last_feat);
  ctx_len_ += n;
}

std::vector<int32_t> DSpark::draft(int32_t anchor, int count) { return run(anchor, count, 0.0f, nullptr, nullptr); }

std::vector<int32_t> DSpark::draft_sample(int32_t anchor, int count, float temperature, const float* uniforms,
                                          std::vector<float>* q_tok) {
  return run(anchor, count, temperature, uniforms, q_tok);
}

int64_t DSpark::vocab() const { return impl_->Vu; }

const float* DSpark::q_row(int j) {
  auto& m = *impl_;
  if (!m.qbuf.valid()) m.qbuf = m.be.alloc(DType::F32, {m.Vu}, "dspark.q");
  SampleParams sp{uint32_t(m.Vu), 0, 0, uint32_t(j), 1.0f / m.last_temp, 0};
  Tensor zr = m.zbuf;
  zr.offset += size_t(j) * m.Vu * 4;
  m.pipe("dspark_qrow");
  m.bind(0, zr);
  m.bind(1, m.stats);
  m.bind(2, m.qbuf);
  m.g.set_value(3, sp);
  m.g.dispatch({cdiv(m.Vu, 256), 1, 1}, {256, 1, 1});
  m.be.sync();
  qrow_.assign(m.qbuf.data<float>(), m.qbuf.data<float>() + m.Vu);
  return qrow_.data();
}

std::vector<int32_t> DSpark::run(int32_t anchor, int count, float temperature, const float* uniforms,
                                 std::vector<float>* q_tok) {
  auto& m = *impl_;
  const bool sampled = temperature > 0.0f && uniforms;
  m.last_temp = temperature;
  count = std::min(count, block_ - 1);
  // D: INK_DSPARK_ROWS=count+1 runs the block with only the rows asked for (anchor + count masks) instead of
  // the full block of 8: fewer rows through the 3 layers and the global attention. The drafts change (the mask
  // rows attend to each other), so this is a speed / acceptance trade to measure.
  static const bool rows_by_count = getenv("INK_DSPARK_ROWS") != nullptr;
  const int64_t B = rows_by_count ? std::min<int64_t>(block_, count + 1) : block_;
  auto rows = [&](const Tensor& t) {
    Tensor v = t;
    v.set_shape({B, t.dim(-1)});
    return v;
  };
  Tensor w_x = rows(m.x);
  Tensor w_xin = rows(m.xin);
  Tensor w_q = rows(m.q);
  Tensor w_kb = rows(m.kb);
  Tensor w_vb = rows(m.vb);
  Tensor w_at = rows(m.at);
  Tensor w_o = rows(m.o);
  Tensor w_summed = rows(m.summed);
  Tensor w_res = rows(m.res);
  Tensor w_post = rows(m.post);
  Tensor w_gt = rows(m.gt);
  Tensor w_ut = rows(m.ut);
  Tensor w_a = rows(m.a);
  Tensor w_h = rows(m.h);
  m.be.write(m.tokens, &anchor, 4);
  Tensor t0 = view(m.tokens, {1});
  m.be.embed(t0, m.embed, m.e0);
  m.be.rms_norm(m.e0, &m.embed_norm, m.eps, m.D, m.e1);
  {
    uint32_t nd[2] = {uint32_t(B), uint32_t(m.D)};
    m.pipe("dspark_block_init");
    m.bind(0, m.e1);
    m.bind(1, m.mask);
    m.bind(2, w_x);
    m.g.set_bytes(3, nd, sizeof(nd));
    m.g.dispatch({cdiv(m.D, 256), uint32_t(B), 1}, {256, 1, 1});
  }
  const int64_t pos0 = ctx_len_;
  for (size_t i = 0; i < m.L.size(); i++) {
    auto& l = m.L[i];
    if (i == 0) {
      m.be.copy(w_x, w_res);
      m.be.rms_norm(w_x, &l.in_ln, m.eps, m.D, w_xin);
    } else {
      m.be.add(w_x, w_res, w_summed);
      m.be.cast(w_summed, w_res);
      m.be.rms_norm(w_summed, &l.in_ln, m.eps, m.D, w_xin);
    }
    m.be.linear(w_xin, l.q, w_q);
    m.be.linear(w_xin, l.k, w_kb);
    m.be.linear(w_xin, l.v, w_vb);
    Tensor q3 = view(w_q, {B, m.H, m.HD}), k3 = view(w_kb, {B, m.KVH, m.HD}), v3 = view(w_vb, {B, m.KVH, m.HD});
    m.be.rms_norm(q3, &l.qn, m.eps, m.HD, q3);
    m.be.rms_norm(k3, &l.kn, m.eps, m.HD, k3);
    m.rope(q3, B, m.H, pos0);
    m.rope(k3, B, m.KVH, pos0);
    m.attend(l, q3, k3, v3, B, w_at);
    m.be.linear(w_at, l.o, w_o);
    m.be.add(w_o, w_res, w_summed);
    m.be.rms_norm(w_summed, &l.post_ln, m.eps, m.D, w_post);
    m.be.linear(w_post, l.gate, w_gt);
    m.be.linear(w_post, l.up, w_ut);
    uint32_t n = uint32_t(B * m.I);
    m.pipe("dspark_silu_mul");
    m.bind(0, w_gt);
    m.bind(1, w_ut);
    m.bind(2, w_a);
    m.g.set_value(3, n);
    m.g.dispatch({cdiv(n, 256), 1, 1}, {256, 1, 1});
    m.be.linear(w_a, l.down, w_x);
    m.be.cast(w_summed, w_res);
  }
  m.be.add(w_x, w_res, w_summed);
  m.be.rms_norm(w_summed, &m.norm, m.eps, m.D, w_h);
  // rows 1..count
  Tensor hc = w_h;
  hc.set_shape({count, m.D});
  hc.offset += size_t(m.D) * 2;
  Tensor base = view(m.base, {count, m.V});
  m.be.linear(hc, m.head, base);
  Tensor lf = m.last_feat;
  lf.set_shape({1, 4, m.D});
  m.mix(lf, 1, 3, m.mk_dflare, m.frnn);
  for (int i = 0; i < count; i++) {
    Tensor dst = m.cat;
    dst.set_shape({1, m.D});
    dst.offset += size_t(i) * 2 * m.D * 2;
    Tensor src = hc;
    src.set_shape({1, m.D});
    src.offset += size_t(i) * m.D * 2;
    m.be.copy(src, dst);
    dst.offset += size_t(m.D) * 2;
    m.be.copy(m.frnn, dst);
  }
  Tensor cat = view(m.cat, {count, 2 * m.D}), ctxm = view(m.ctxm, {count, 3 * m.R});
  m.be.linear(cat, m.wc, ctxm);
  m.be.copy(m.zero_state, m.state);
  uint32_t nparts = cdiv(m.Vu, 32);
  for (int i = 0; i < count; i++) {
    MarkovParams mp{uint32_t(m.R), uint32_t(i)};
    m.pipe("dspark_markov_step");
    m.bind(0, m.state);
    m.bind(1, m.latent);
    Tensor cr = m.ctxm;
    cr.offset += size_t(i) * 3 * m.R * 4;
    m.bind(2, cr);
    m.bind(3, m.ws);
    m.bind(4, m.w1);
    m.bind(5, m.tokens);
    m.g.set_value(6, mp);
    m.bind(7, m.rnn_b);
    m.g.dispatch({1, 1, 1}, {256, 1, 1});
    FoldParams fp{uint32_t(m.Vu), uint32_t(m.R), m.cap, sampled ? 1u : 0u};
    m.pipe("dspark_fold_argmax");
    m.bind(0, m.latent);
    m.bind(1, m.fold_w);
    m.bind(2, m.fold_b);
    Tensor br = m.base;
    br.offset += size_t(i) * m.V * 2;
    m.bind(3, br);
    m.bind(4, m.part);
    m.g.set_value(5, fp);
    Tensor zr = m.zbuf;
    zr.offset += size_t(i) * m.Vu * 4;
    m.bind(6, zr);
    m.g.dispatch({nparts, 1, 1}, {256, 1, 1});
    if (sampled) {
      SampleParams sp{uint32_t(m.Vu), nparts, uint32_t(i + 1), uint32_t(i), 1.0f / temperature, uniforms[i]};
      m.pipe("dspark_sample_final");
      m.bind(0, zr);
      m.bind(1, m.part);
      m.bind(2, m.tokens);
      m.bind(3, m.stats);
      m.g.set_value(4, sp);
      m.g.dispatch({1, 1, 1}, {1024, 1, 1});
    } else {
      ArgmaxParams ap{nparts, uint32_t(i + 1)};
      m.pipe("dspark_argmax_final");
      m.bind(0, m.part);
      m.bind(1, m.tokens);
      m.g.set_value(2, ap);
      m.g.dispatch({1, 1, 1}, {1024, 1, 1});
    }
  }
  m.be.sync();
  std::vector<int32_t> out(size_t(count + 1));
  memcpy(out.data(), m.tokens.data<int32_t>(), out.size() * 4);
  out.erase(out.begin());
  if (sampled && q_tok) {
    q_tok->resize(size_t(count));
    for (int i = 0; i < count; i++) (*q_tok)[i] = m.stats.data<float>()[i * 4 + 2];
  }
  return out;
}

}  // namespace ink
