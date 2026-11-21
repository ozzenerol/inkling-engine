// Owner: A. Forward pass of Inkling 2 Mini (mirrors mlx_vlm/models/inkling2/language.py).
#include "model/model.h"

#include <cstdlib>
#include <cstring>

#include "core/bf16.h"
#include "core/parallel.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace ink {

namespace {
const std::string P = "language_model.model.";
}

Model::Model(Backend& be, const std::string& dir, int64_t max_context, int64_t max_chunk)
    : be_(be), cfg_(ModelConfig::load(dir)), max_ctx_(max_context), max_chunk_(max_chunk) {
  st_.add_dir(dir);
  load(dir);
}

Model::~Model() {
  auto fr = [&](Tensor& t) { if (t.valid()) be_.free(t); };
  for (auto& t : owned_) fr(t);
  for (auto& v : vs_) {
    fr(v.kv.k);
    fr(v.kv.v);
    for (auto& c : v.conv) fr(c);
    for (auto& c : v.hist) fr(c);
  }
  for (Tensor* t : {&ids_, &x_, &x2_, &prelude_, &blk_[0], &blk_[1], &blk_[2], &blk_[3], &bank_[0], &bank_[1], &msg_,
                    &mixed_, &parts_[0], &parts_[1], &parts_[2], &partsb_[0], &partsb_[1], &partsb_[2], &kraw_,
                    &vraw_, &kc_, &vc_, &dk_, &dv_, &q_, &k_, &v_, &r_, &sink_, &attn_, &gate_, &ao_, &res_, &resb_,
                    &hid_, &hn_, &ridx_, &rw_, &moe_, &mo_, &h_, &hfin_, &tau_, &logits_dev_})
    fr(*t);
}

// Two checkpoint layouts are supported:
//  - netprojects 4-8bit (mlx-vlm names): dense weights bf16, experts gate_proj / up_proj / down_proj.
//  - pipenetwork 8bit (mlx-lm names): dense weights MLX affine 8-bit (w = scale * q + bias per group of 64),
//    experts gate_up_proj fused (rows interleaved gate, up, gate, ...), router and controller gates in f32.
// Stage 1 for the 8-bit layout: everything is converted at load into the first layout's in-memory form (affine
// dequantized to bf16, f32 -> bf16 where the values are bf16 anyway, experts split), so all kernels run as they
// are. Stage 2 (B): kernels that read the 8-bit codes in place, which is where the speed comes from.
void Model::load(const std::string&) {
  const bool alt = st_.find("model.language_model.embed_norm.weight") != nullptr;
  alt_layout_ = alt;
  auto rep = [](std::string& n, const std::string& a, const std::string& b) {
    size_t at = n.find(a);
    if (at != std::string::npos) n.replace(at, a.size(), b);
  };
  auto name = [&](std::string n) {
    if (!alt) return n;
    rep(n, "language_model.model.", "model.language_model.");
    rep(n, "language_model.lm_head.", "lm_head.");
    rep(n, "message_queries", "message_input.query");
    rep(n, ".attn_sconv.conv.weight", ".post_attention_convolution.conv1d.weight");
    rep(n, ".mlp_sconv.conv.weight", ".post_mlp_convolution.conv1d.weight");
    rep(n, "_sconv.conv.weight", "_sconv.conv1d.weight");
    rep(n, "self_attn.q_scale", "self_attn.q_l2_clip.scale");
    rep(n, "self_attn.rel_scale", "self_attn.rel_l2_clip.scale");
    rep(n, "self_attn.rel_proj", "self_attn.rel_logits_proj.proj");
    rep(n, ".controllers.attn.", ".q_message_gate.");
    rep(n, ".controllers.k.", ".k_message_gate.");
    rep(n, ".controllers.v.", ".v_message_gate.");
    rep(n, ".controllers.mlp.", ".mlp_message_gate.");
    rep(n, "down_weight", "down_proj.weight");
    rep(n, "forget_weight", "forget_gate_proj.weight");
    rep(n, "gain_weight", "message_gate_proj.weight");
    rep(n, "mlp.gate_weight", "mlp.gate.weight");
    rep(n, "mlp.e_score_correction_bias", "mlp.gate.e_score_correction_bias");
    return n;
  };
  auto own = [&](DType dt, std::initializer_list<int64_t> shape, const void* data, size_t bytes) {
    Tensor t = be_.alloc(dt, shape, "weight");
    be_.write(t, data, bytes);
    owned_.push_back(t);
    return t;
  };
  auto raw = [&](const SafeTensors::Entry& e) {
    std::vector<uint8_t> v(e.bytes);
    read_entry(e, v.data());
    return v;
  };
  long inexact = 0;
  // INK_Q8=0 forces the load-time conversion even when the backend has 8-bit kernels (A/B checks).
  const char* q8env = std::getenv("INK_Q8");
  const bool q8_inplace = be_.supports_q8() && !(q8env && q8env[0] == '0');
  // Dense weight in canonical form: bf16 [N,K] (or the stored dtype when it is not bf16-like).
  auto W = [&](const std::string& canon) -> Tensor {
    const std::string n = name(canon);
    const auto& e = st_.at(n);
    const std::string base = n.size() > 7 && n.compare(n.size() - 7, 7, ".weight") == 0 ? n.substr(0, n.size() - 7) : n;
    const auto* es = st_.find(base + ".scales");
    const auto* eb = st_.find(base + ".biases");
    if (e.dtype == DType::U32 && es && eb && es->dtype == DType::BF16) {
      if (q8_inplace) {
        Tensor codes = be_.load_weight(e);
        be_.bind_affine(codes, be_.load_weight(*es), be_.load_weight(*eb));
        return codes;
      }
      // MLX affine 8-bit: 4 codes per u32 (low byte first), group 64.
      const int64_t N = e.shape[0], K = e.shape[1] * 4, G = es->shape[1];
      if (G * 64 != K) throw std::runtime_error("affine group size != 64: " + n);
      auto codes = raw(e), sc = raw(*es), bi = raw(*eb);
      std::vector<uint16_t> out(static_cast<size_t>(N * K));
      const uint16_t* scp = reinterpret_cast<const uint16_t*>(sc.data());
      const uint16_t* bip = reinterpret_cast<const uint16_t*>(bi.data());
      parallel_for(N, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r)
          for (int64_t g = 0; g < G; ++g) {
            const float sv = bf16_to_f32(scp[r * G + g]), bv = bf16_to_f32(bip[r * G + g]);
            const uint8_t* c = codes.data() + (r * K + g * 64);
            uint16_t* o = out.data() + r * K + g * 64;
            for (int j = 0; j < 64; ++j) o[j] = f32_to_bf16(sv * float(c[j]) + bv);
          }
      }, 16);
      return own(DType::BF16, {N, K}, out.data(), out.size() * 2);
    }
    return be_.load_weight(e);
  };
  // Like W, but f32 storage is converted to bf16 (router and controller gates; the values are bf16 already).
  auto Wbf = [&](const std::string& canon) -> Tensor {
    const auto& e = st_.at(name(canon));
    if (e.dtype != DType::F32) return W(canon);
    auto v = raw(e);
    const float* f = reinterpret_cast<const float*>(v.data());
    const size_t n = v.size() / 4;
    std::vector<uint16_t> out(n);
    for (size_t i = 0; i < n; ++i) {
      out[i] = f32_to_bf16(f[i]);
      inexact += bf16_to_f32(out[i]) != f[i];
    }
    Tensor t = be_.alloc(DType::BF16, {e.shape[0], e.shape[1]}, "weight");
    be_.write(t, out.data(), n * 2);
    owned_.push_back(t);
    return t;
  };
  // One half (part 0 = gate, 1 = up) of a fused, row-interleaved gate_up tensor [E, 2I, W] -> [E, I, W].
  auto split = [&](const SafeTensors::Entry& e, int part) -> Tensor {
    const int64_t E = e.shape[0], I2 = e.shape[1], Wd = e.shape[2];
    const size_t rb = size_t(Wd) * dtype_size(e.dtype);
    auto v = raw(e);
    std::vector<uint8_t> out(size_t(E) * (I2 / 2) * rb);
    for (int64_t x = 0; x < E; ++x)
      for (int64_t i = 0; i < I2 / 2; ++i)
        std::memcpy(&out[(x * (I2 / 2) + i) * rb], &v[(x * I2 + 2 * i + part) * rb], rb);
    return own(e.dtype, {E, I2 / 2, Wd}, out.data(), out.size());
  };
  embed_ = W(P + "embed_tokens.weight");
  embed_norm_ = W(P + "embed_norm.weight");
  final_query_ = W(P + "final_input.query");
  final_norm_ = W(P + "norm.weight");
  head_ = W("language_model.lm_head.weight");
  msg_queries_ = W(P + "message_queries");
  const int D = cfg_.hidden;
  lw_.resize(cfg_.layers);
  for (int L = 0; L < cfg_.layers; ++L) {
    auto& w = lw_[L];
    const std::string p = P + "layers." + std::to_string(L) + ".";
    w.local = cfg_.is_local[L];
    if (L > 0) w.attn_query = W(p + "attention_input.query");
    w.mlp_query = W(p + "mlp_input.query");
    if (w.local) {
      w.in_norm = W(p + "input_layernorm.weight");
    } else {
      w.q_norm = W(p + "q_input_layernorm.weight");
      w.k_norm = W(p + "k_input_layernorm.weight");
      w.v_norm = W(p + "v_input_layernorm.weight");
    }
    w.post_norm = W(p + "post_attention_layernorm.weight");
    const std::string a = p + "self_attn.";
    w.q_proj = W(a + "q_proj.weight");
    w.k_proj = W(a + "k_proj.weight");
    w.v_proj = W(a + "v_proj.weight");
    w.r_proj = W(a + "r_proj.weight");
    w.o_proj = W(a + "o_proj.weight");
    w.gate_proj = W(a + "gate_proj.weight");
    w.sink_proj = W(a + "sink_proj.weight");
    w.q_scale = W(a + "q_scale");
    w.rel_scale = W(a + "rel_scale");
    w.rel_proj = W(a + "rel_proj");
    if (w.local) {
      w.k_conv = W(a + "k_sconv.conv.weight");
      w.v_conv = W(a + "v_sconv.conv.weight");
    } else {
      w.k_static = W(a + "k_sconv.static_weight");
      w.v_static = W(a + "v_sconv.static_weight");
      w.k_wproj = W(a + "k_sconv.weight_proj.weight");
      w.v_wproj = W(a + "v_sconv.weight_proj.weight");
    }
    w.attn_sconv = W(p + "attn_sconv.conv.weight");
    w.mlp_sconv = W(p + "mlp_sconv.conv.weight");
    if (L >= cfg_.core_start && L < cfg_.core_end) {
      w.has_ctrl = true;
      const char* names[4] = {"attn", "k", "v", "mlp"};
      for (int i = 0; i < 4; ++i) {
        const std::string c = p + "controllers." + names[i] + ".";
        w.ctrl_down[i] = Wbf(c + "down_weight");
        w.ctrl_forget[i] = Wbf(c + "forget_weight");
        w.ctrl_gain[i] = Wbf(c + "gain_weight");
      }
    }
    const std::string m = p + "mlp.";
    w.moe.gate_w = Wbf(m + "gate_weight");
    w.moe.bias = W(m + "e_score_correction_bias");
    QuantProj* projs[3] = {&w.moe.gate, &w.moe.up, &w.moe.down};
    const char* pn[3] = {"gate_proj", "up_proj", "down_proj"};
    for (int i = 0; i < 3; ++i) {
      QuantProj& qp = *projs[i];
      if (alt) {
        const std::string q = name(m) + "experts." + (i < 2 ? "gate_up_proj" : "down_proj") + ".";
        const auto* eg = st_.find(q + "global_scale");
        if (i < 2) {
          qp.weight = split(st_.at(q + "weight"), i);
          qp.scales = split(st_.at(q + "scales"), i);
        } else {
          qp.weight = be_.load_weight(st_.at(q + "weight"));
          qp.scales = be_.load_weight(st_.at(q + "scales"));
        }
        if (eg) {
          // This layout stores MLX's kernel factor 2688 * weight_scale_2; ours is weight_scale_2 (the division
          // gives back the 4-8bit build's values bit for bit: tests/ref/check_8bit_layout.py).
          auto v = raw(*eg);
          float* f = reinterpret_cast<float*>(v.data());
          for (int64_t x = 0; x < eg->shape[0]; ++x) f[x] = f[x] / 2688.0f;
          qp.global = own(DType::F32, {eg->shape[0]}, v.data(), v.size());
        }
      } else {
        const std::string q = m + "switch_mlp." + pn[i] + ".";
        qp.weight = be_.load_weight(st_.at(q + "weight"));
        qp.scales = be_.load_weight(st_.at(q + "scales"));
        if (auto* eg = st_.find(q + "weight_scale_2")) qp.global = be_.load_weight(*eg);
      }
      qp.mode = qp.global.valid() ? QuantMode::NVFP4 : QuantMode::MXFP8;
      qp.n = qp.weight.dim(1);
      qp.k = qp.weight.dim(2) * (qp.mode == QuantMode::NVFP4 ? 8 : 4);
      int64_t group = qp.mode == QuantMode::NVFP4 ? 16 : 32;
      if (qp.scales.dim(2) * group != qp.k) throw std::runtime_error("bad quant shapes, layer " + std::to_string(L));
    }
    (void)D;
  }

  if (inexact) std::fprintf(stderr, "[model] warning: %ld f32 weights were not exact in bf16\n", inexact);

  // Per-visit state.
  vs_.resize(cfg_.num_visits());
  const int64_t local_cap = cfg_.window + 512;
  for (int v = 0; v < cfg_.num_visits(); ++v) {
    const int L = cfg_.visits[v];
    auto& s = vs_[v];
    s.kv.ring = cfg_.is_local[L];
    s.kv.cap = s.kv.ring ? local_cap : max_ctx_;
    s.kv.k = be_.alloc(DType::BF16, {cfg_.kv_heads, s.kv.cap, cfg_.head_dim}, "kv.k");
    s.kv.v = be_.alloc(DType::BF16, {cfg_.kv_heads, s.kv.cap, cfg_.head_dim}, "kv.v");
    const int64_t kvc = int64_t(cfg_.kv_heads) * cfg_.head_dim;
    s.conv[0] = be_.alloc(DType::F32, {3, kvc}, "conv.k");
    s.conv[1] = be_.alloc(DType::F32, {3, kvc}, "conv.v");
    s.conv[2] = be_.alloc(DType::F32, {3, cfg_.hidden}, "conv.attn");
    s.conv[3] = be_.alloc(DType::F32, {3, cfg_.hidden}, "conv.mlp");
  }

  // Workspace.
  const int64_t T = max_chunk_, Dm = cfg_.hidden, H = cfg_.heads, HD = cfg_.head_dim, KVC = cfg_.kv_heads * HD;
  auto A = [&](DType d, std::initializer_list<int64_t> s, const char* tag) { return be_.alloc(d, s, tag); };
  ids_ = A(DType::I32, {T}, "ids");
  x_ = A(DType::BF16, {T, Dm}, "x");
  x2_ = A(DType::BF16, {T, Dm}, "x2");
  prelude_ = A(DType::BF16, {T, Dm}, "prelude");
  for (auto& b : blk_) b = A(DType::BF16, {T, Dm}, "block");
  for (auto& b : bank_) b = A(DType::BF16, {T, Dm}, "bank");
  msg_ = A(DType::F32, {4, T, Dm}, "msg");
  mixed_ = A(DType::F32, {3, T, Dm}, "mixed");
  for (auto& p : parts_) p = A(DType::F32, {T, Dm}, "part");
  for (auto& p : partsb_) p = A(DType::BF16, {T, Dm}, "partb");
  kraw_ = A(DType::BF16, {T, KVC}, "kraw");
  vraw_ = A(DType::BF16, {T, KVC}, "vraw");
  kc_ = A(DType::BF16, {T, KVC}, "kc");
  vc_ = A(DType::BF16, {T, KVC}, "vc");
  dk_ = A(DType::BF16, {T, cfg_.kv_heads * 4}, "dk");
  dv_ = A(DType::BF16, {T, cfg_.kv_heads * 4}, "dv");
  q_ = A(DType::BF16, {T, H * HD}, "q");
  k_ = A(DType::BF16, {T, KVC}, "k");
  v_ = A(DType::BF16, {T, KVC}, "v");
  r_ = A(DType::BF16, {T, H * cfg_.d_rel}, "r");
  sink_ = A(DType::BF16, {T, H}, "sink");
  attn_ = A(DType::BF16, {T, H * HD}, "attn");
  gate_ = A(DType::BF16, {T, H * HD}, "gate");
  ao_ = A(DType::BF16, {T, Dm}, "ao");
  res_ = A(DType::F32, {T, Dm}, "res");
  resb_ = A(DType::BF16, {T, Dm}, "resb");
  hid_ = A(DType::F32, {T, Dm}, "hid");
  hn_ = A(DType::BF16, {T, Dm}, "hn");
  ridx_ = A(DType::I32, {T, cfg_.topk}, "ridx");
  rw_ = A(DType::F32, {T, cfg_.topk}, "rw");
  moe_ = A(DType::BF16, {T, Dm}, "moe");
  mo_ = A(DType::BF16, {T, Dm}, "mo");
  h_ = A(DType::F32, {T, Dm}, "h");
  hfin_ = A(DType::F32, {T, Dm}, "hfin");
  tau_ = A(DType::F32, {T}, "tau");
  logits_dev_ = A(DType::F32, {T, cfg_.vocab_unpadded}, "logits");
}

Tensor Model::view(const Tensor& t, std::initializer_list<int64_t> shape) const {
  Tensor v = t;
  v.set_shape(shape);
  if (v.nbytes() > t.nbytes()) throw std::runtime_error("view larger than tensor " + t.str());
  return v;
}

Tensor Model::row_view(const Tensor& t, int64_t first, std::initializer_list<int64_t> shape) const {
  Tensor v = t;
  v.set_shape(shape);
  v.offset = t.offset + size_t(first) * v.nbytes();
  return v;
}

void Model::reset() {
  len_ = 0;
  toks_.clear();
  for (auto& s : vs_) {
    s.kv.len = 0;
    for (auto& c : s.conv) {
      std::vector<char> z(c.nbytes(), 0);
      be_.write(c, z.data(), z.size());
    }
  }
}

void Model::set_len(int64_t n) {
  len_ = n;
  toks_.resize(static_cast<size_t>(n), -1);
  for (auto& s : vs_) s.kv.len = n;
}

void Model::write_tau(int64_t T) {
  // Global-layer tau (log scaling) for positions len+1 .. len+T.
  std::vector<float> tau(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    float pos = float(len_ + i + 1);
    tau[i] = 1.0f + cfg_.log_alpha * std::log(std::max(pos / float(cfg_.log_n_floor), 1.0f));
  }
  Tensor tt = view(tau_, {T});
  be_.write(tt, tau.data(), size_t(T) * 4);
}

// One layer visit: x (bf16 [T,D]) is updated in place. blocks = current attention-residual blocks (appended to at
// block boundaries, storage blk_[blocks.size()]). msg = loop messages f32 [4,T,D] or invalid.
void Model::run_visit(int v, Tensor& x, std::vector<Tensor>& blocks, const ChainMsg* cm, int64_t T) {
  auto dbg = [&](const char* n, const Tensor& t) { if (on_debug) on_debug(v, n, t); };
  const int L = cfg_.visits[v];
  const int64_t D = cfg_.hidden;
  const auto& w = lw_[L];
  const bool boundary = L % cfg_.block == 0;
  const int nq = w.local ? 1 : 3;
  // Chain kernels (A): one dispatch for the front (mix, controllers, norms) and one for the middle (conv, mix,
  // controller, norm, router). Decode and verify only; the debug hook needs the separate intermediates.
  const bool chain = be_.has_chain(T) && !on_debug && !no_chain_;
  const bool chain_front = chain && L > 0;
  ChainCtrl ctrl_w[4];
  if (cm)
    for (int i = 0; i < 4; ++i) ctrl_w[i] = ChainCtrl{w.ctrl_down[i], w.ctrl_forget[i], w.ctrl_gain[i]};
  // The message tensor is only materialized for paths that are not chained.
  Tensor msg;
  if (cm && !(chain && chain_front)) {
    msg = view(msg_, {4, T, D});
    be_.loop_messages({&cm->bank0, &cm->bank1}, cm->queries, msg);
  }
  const bool has_msg = cm != nullptr;
  Tensor xq, xk, xv;
  if (chain_front) {
    ChainFrontArgs fa;
    fa.has_conv = pending_.on;
    if (pending_.on) {
      fa.conv = pending_.conv;
      fa.res = pending_.res;
      pending_.on = false;
    }
    fa.x = x;
    for (auto& b : blocks) fa.blocks.push_back(&b);
    fa.nq = nq;
    fa.query = w.attn_query;
    fa.gamma = w.local ? w.in_norm : w.q_norm;
    fa.gamma_k = w.k_norm;
    fa.gamma_v = w.v_norm;
    fa.msg = cm;
    fa.ctrl = cm ? ctrl_w : nullptr;
    fa.eps = cfg_.rms_eps;
    fa.xq = view(partsb_[0], {T, D});
    const bool three = cm || nq == 3;
    fa.xk = three ? view(partsb_[1], {T, D}) : fa.xq;
    fa.xv = three ? view(partsb_[2], {T, D}) : fa.xq;
    be_.chain_front(fa);
    xq = fa.xq;
    xk = fa.xk;
    xv = fa.xv;
  }
  if (pending_.on) throw std::runtime_error("pending conv was not consumed");
  if (on_visit) on_visit(v, x);  // x is complete here (after a deferred tail), and unchanged until the visit ends
  const bool fused_in = L > 0 && w.local && !has_msg;  // mix + input norm in one op (B)
  if (L > 0 && !chain_front) {
    Tensor mixed = view(mixed_, {nq, T, D});
    std::vector<const Tensor*> states;
    for (auto& b : blocks) states.push_back(&b);
    states.push_back(&x);
    if (fused_in) {
      Tensor xqn = view(partsb_[0], {T, D});
      be_.attn_residual_norm(states, w.attn_query, mixed, w.in_norm, cfg_.rms_eps, xqn);
    } else {
      be_.attn_residual(states, w.attn_query, mixed);
    }
    dbg("mixed", mixed);
  }
  auto part = [&](int i) -> Tensor {
    if (L == 0) return x;
    return row_view(mixed_, std::min(i, nq - 1), {T, D});
  };
  if (chain_front) {
    // done above
  } else if (w.local && !has_msg) {
    xq = view(partsb_[0], {T, D});
    if (!fused_in) {
      Tensor p0 = part(0);
      be_.rms_norm(p0, &w.in_norm, cfg_.rms_eps, D, xq);
    }
    xk = xv = xq;
  } else {
    Tensor ins[3];
    for (int i = 0; i < 3; ++i) ins[i] = part(i);
    if (has_msg) {
      // the three controllers (attn, k, v parts) in one batched op (B); same order and math as three calls
      Tensor outs_c[3], ms_c[3];
      for (int i = 0; i < 3; ++i) {
        outs_c[i] = view(parts_[i], {T, D});
        ms_c[i] = row_view(msg, i, {T, D});
      }
      be_.controller_multi(3, ins, ms_c, w.ctrl_down.data(), w.ctrl_forget.data(), w.ctrl_gain.data(), outs_c);
      dbg("ctrl0", outs_c[0]);
      for (int i = 0; i < 3; ++i) ins[i] = outs_c[i];
    }
    const Tensor* norms[3] = {w.local ? &w.in_norm : &w.q_norm, w.local ? &w.in_norm : &w.k_norm,
                              w.local ? &w.in_norm : &w.v_norm};
    Tensor outs[3];
    for (int i = 0; i < 3; ++i) outs[i] = view(partsb_[i], {T, D});
    be_.rms_norm_multi(3, ins, norms, cfg_.rms_eps, D, outs);
    dbg("xq", outs[0]);
    dbg("xk", outs[1]);
    dbg("xv", outs[2]);
    if (has_msg) dbg("msg", msg);
    xq = outs[0];
    xk = outs[1];
    xv = outs[2];
  }

  // ---- attention ----
  auto& s = vs_[v];
  const int64_t H = cfg_.heads, HD = cfg_.head_dim, KVH = cfg_.kv_heads, KVC = KVH * HD;
  Tensor kraw = view(kraw_, {T, KVC}), vraw = view(vraw_, {T, KVC});
  Tensor q = view(q_, {T, H * HD}), r = view(r_, {T, H * cfg_.d_rel}), sink = view(sink_, {T, H});
  Tensor g = view(gate_, {T, H * HD});
  Tensor dk = view(dk_, {T, KVH * 4}), dv = view(dv_, {T, KVH * 4});
  if (w.local && !has_msg) {
    // One shared input: q, k, v, r, sink, gate in one multi-linear.
    be_.linear_multi(xq, {&w.q_proj, &w.k_proj, &w.v_proj, &w.r_proj, &w.sink_proj, &w.gate_proj},
                     {&q, &kraw, &vraw, &r, &sink, &g});
  } else {
    be_.linear_multi(xq, {&w.q_proj, &w.r_proj, &w.sink_proj, &w.gate_proj}, {&q, &r, &sink, &g});
    if (w.local) {
      be_.linear(xk, w.k_proj, kraw);
      be_.linear(xv, w.v_proj, vraw);
    } else {
      be_.linear_multi(xk, {&w.k_proj, &w.k_wproj}, {&kraw, &dk});
      be_.linear_multi(xv, {&w.v_proj, &w.v_wproj}, {&vraw, &dv});
    }
  }
  dbg("q_raw", q);
  dbg("k_raw", kraw);
  dbg("v_raw", vraw);
  Tensor kc = view(kc_, {T, KVC}), vc = view(vc_, {T, KVC});
  Tensor* hist[4] = {nullptr, nullptr, nullptr, nullptr};
  if (keep_hist_)
    for (int i = 0; i < 4; ++i) {
      int64_t C = s.conv[i].dim(1);
      hist_views_[i] = row_view(s.hist[i], 0, {T, 3, C});
      hist[i] = &hist_views_[i];
    }
  Tensor tau = view(tau_, {T});
  Tensor q3 = view(q_, {T, H, HD});
  Tensor k3 = view(k_, {T, KVH, HD}), kc3 = view(kc_, {T, KVH, HD});
  Tensor v3 = view(v_, {T, KVH, HD});
  Tensor r3 = view(r_, {T, H, cfg_.d_rel});
  if (w.local) {
    // fused (B): short convs, l2_clip q/k/r, rms v -- same order and math as the primitive sequence
    be_.attn_prep_local(kraw, vraw, w.k_conv, w.v_conv, s.conv[0], s.conv[1], hist[0], hist[1], q3, w.q_scale, r3,
                        w.rel_scale, cfg_.qk_clip, cfg_.rel_clip, cfg_.rms_eps, kc, vc, k3, v3);
  } else {
    be_.dyn_conv(kraw, dk, w.k_static, s.conv[0], kc, hist[0]);
    be_.dyn_conv(vraw, dv, w.v_static, s.conv[1], vc, hist[1]);
    be_.l2_clip(q3, &w.q_scale, cfg_.qk_clip, &tau, q3);
    be_.l2_clip(kc3, nullptr, cfg_.qk_clip, nullptr, k3);
    be_.rms_norm(vc, nullptr, cfg_.rms_eps, HD, v3);
    be_.l2_clip(r3, &w.rel_scale, cfg_.rel_clip, &tau, r3);
  }
  dbg("q", q3);
  dbg("k", k3);
  dbg("v", v3);
  dbg("r", r3);
  dbg("sink", sink);
  AttnArgs aa;
  aa.q = q3;
  aa.k_new = k3;
  aa.v_new = v3;
  aa.r = r3;
  aa.rel_proj = w.rel_proj;
  aa.sink = sink;
  aa.kv = s.kv;
  aa.window = w.local ? cfg_.window : 0;
  Tensor at = view(attn_, {T, H * HD});
  be_.attention_gated(aa, g, at);  // attention + sigmoid gate (fused on Metal)
  dbg("attn", at);
  be_.kv_append(k3, v3, s.kv);
  Tensor ao = view(ao_, {T, D});
  be_.linear(at, w.o_proj, ao);
  dbg("o", ao);

  // ---- residual + MoE ----
  Tensor resb = view(resb_, {T, D}), res = view(res_, {T, D});
  Tensor hn = view(hn_, {T, D});
  Tensor ridx = view(ridx_, {T, cfg_.topk}), rw = view(rw_, {T, cfg_.topk});
  if (chain) {
    ChainMidArgs ma;
    ma.conv = ChainConv{ao, w.attn_sconv, &s.conv[2], hist[2]};
    ma.x = x;
    ma.boundary = boundary;
    const size_t nold = blocks.size();
    if (boundary) {
      ma.blk = view(blk_[nold], {T, D});
      blocks.push_back(ma.blk);  // may reallocate: take the pointers afterwards
    }
    for (size_t b = 0; b < nold; ++b) ma.blocks.push_back(&blocks[b]);
    ma.query = w.mlp_query;
    ma.gamma = w.post_norm;
    ma.msg = cm;
    ma.ctrl = cm ? &ctrl_w[3] : nullptr;
    ma.moe = &w.moe;
    ma.topk = cfg_.topk;
    ma.route_scale = cfg_.route_scale;
    ma.eps = cfg_.rms_eps;
    ma.res = res;
    ma.hn = hn;
    ma.idx = ridx;
    ma.w = rw;
    be_.chain_mid(ma);
  } else {
    if (boundary) {
      be_.short_conv(ao, w.attn_sconv, s.conv[2], resb, hist[2]);
      Tensor blk = view(blk_[blocks.size()], {T, D});
      be_.copy(x, blk);
      blocks.push_back(blk);
      be_.cast(resb, res);
    } else if (has_msg) {
      be_.conv_add(ao, w.attn_sconv, s.conv[2], hist[2], x, resb, res);  // conv + residual add (fused on Metal)
    }
    Tensor hid = view(hid_, {1, T, D});
    {
      std::vector<const Tensor*> states;
      for (auto& b : blocks) states.push_back(&b);
      states.push_back(&res);
      if (!has_msg && !boundary) {
        // conv + residual add + mix + post norm: one op (B); default = conv_add then attn_residual_norm
        Tensor hn0 = view(hn_, {T, D});
        be_.conv_add_mix_norm(ao, w.attn_sconv, s.conv[2], hist[2], x, resb, res, states, w.mlp_query, hid,
                              w.post_norm, cfg_.rms_eps, hn0);
      } else if (!has_msg) {
        Tensor hn0 = view(hn_, {T, D});
        be_.attn_residual_norm(states, w.mlp_query, hid, w.post_norm, cfg_.rms_eps, hn0);  // mix + post norm (B)
      } else {
        be_.attn_residual(states, w.mlp_query, hid);
      }
    }
    dbg("hid_mix", hid);
    Tensor hid2 = view(hid_, {T, D});
    if (has_msg) {
      Tensor m3 = row_view(msg, 3, {T, D});
      be_.controller(hid2, m3, w.ctrl_down[3], w.ctrl_forget[3], w.ctrl_gain[3], hid2);
      be_.rms_norm(hid2, &w.post_norm, cfg_.rms_eps, D, hn);
    }
    dbg("hn", hn);
    be_.router(hn, w.moe, cfg_.topk, cfg_.route_scale, ridx, rw);
  }
  Tensor mo = view(moe_, {T, D});
  be_.moe(hn, ridx, rw, w.moe, mo);
  dbg("moe", mo);
  Tensor mc = view(mo_, {T, D});
  // x = res + conv(mo). With chains the next local visit's front kernel does it (same pass, same block list).
  const int Ln = v + 1 < cfg_.num_visits() ? cfg_.visits[v + 1] : -1;
  if (chain && Ln > 0 && Ln != cfg_.core_start) {
    pending_.on = true;
    pending_.res = res;
    pending_.conv = ChainConv{mo, w.mlp_sconv, &s.conv[3], nullptr};
    if (hist[3]) {
      pending_.hist_view = *hist[3];
      pending_.conv.hist = &pending_.hist_view;
    }
    return;
  }
  be_.conv_add(mo, w.mlp_sconv, s.conv[3], hist[3], res, mc, x);  // x = res + conv(mo) (bf16; fp32 add commutes)
  dbg("res", res);
  dbg("out", x);
}

void Model::forward(const int32_t* ids, int64_t T, bool all_logits, bool keep_hist) {
  if (T <= 0 || T > max_chunk_) throw std::runtime_error("forward: bad T");
  if (len_ + T > max_ctx_) throw std::runtime_error("forward: context full");
  if (keep_hist && T > kMaxHist) throw std::runtime_error("forward: keep_hist needs T <= 16");
  if (keep_hist) ensure_hist();
  keep_hist_ = keep_hist;
  hist_rows_ = keep_hist ? T : 0;
  const int64_t D = cfg_.hidden;
  Tensor idt = view(ids_, {T});
  be_.write(idt, ids, size_t(T) * 4);
  write_tau(T);

  Tensor x = view(x_, {T, D}), xn = view(x2_, {T, D});
  be_.embed(idt, embed_, xn);
  be_.rms_norm(xn, &embed_norm_, cfg_.rms_eps, D, x);

  std::vector<Tensor> blocks;  // views of blk_ in use
  int prelude_n = -1;
  bool have_bank = false;
  for (int v = 0; v < cfg_.num_visits(); ++v) {
    const int L = cfg_.visits[v];
    if (L == cfg_.core_start) {
      if (prelude_n < 0) {
        Tensor pre = view(prelude_, {T, D});
        be_.copy(x, pre);
        prelude_n = int(blocks.size());
      } else {
        // bank = completed core blocks (after the prelude partial) + current partial
        int nb = 0;
        for (size_t b = size_t(prelude_n) + 1; b < blocks.size(); ++b) {
          Tensor bk = view(bank_[nb++], {T, D});
          be_.copy(blocks[b], bk);
        }
        Tensor bk = view(bank_[nb++], {T, D});
        be_.copy(x, bk);
        if (nb != 2) throw std::runtime_error("unexpected bank size");
        have_bank = true;
        be_.copy(view(prelude_, {T, D}), x);
        blocks.resize(size_t(prelude_n));
      }
    }
    ChainMsg cmsg;
    const bool with_msg = have_bank && L < cfg_.core_end;
    if (with_msg) {
      cmsg.bank0 = view(bank_[0], {T, D});
      cmsg.bank1 = view(bank_[1], {T, D});
      cmsg.queries = row_view(msg_queries_, L - cfg_.core_start, {4, D});
    }
    run_visit(v, x, blocks, with_msg ? &cmsg : nullptr, T);  // calls on_visit(v, x) once x is complete
  }

  // Final: attention-residual readout, norm, logits.
  Tensor h = view(h_, {1, T, D});
  {
    std::vector<const Tensor*> states;
    for (auto& b : blocks) states.push_back(&b);
    states.push_back(&x);
    be_.attn_residual(states, final_query_, h);
  }
  Tensor h2 = view(h_, {T, D}), hf = view(hfin_, {T, D});
  be_.rms_norm(h2, &final_norm_, cfg_.rms_eps, D, hf);
  if (on_visit) on_visit(cfg_.num_visits(), hf);
  const int64_t V = cfg_.vocab_unpadded;
  const int64_t rows = all_logits ? T : 1;
  Tensor hrow = all_logits ? hf : row_view(hfin_, T - 1, {1, D});
  Tensor lg = view(logits_dev_, {rows, V});
  be_.logits(hrow, head_, V, cfg_.logit_cap, lg);
  be_.sync();
  logits_.resize(static_cast<size_t>(rows * V));
  be_.read(lg, logits_.data(), logits_.size() * 4);
  logit_rows_ = rows;
  toks_.resize(static_cast<size_t>(len_), -1);
  toks_.insert(toks_.end(), ids, ids + T);
  len_ += T;
  keep_hist_ = false;
}

void Model::ensure_hist() {
  if (vs_.empty() || vs_[0].hist[0].valid()) return;
  for (auto& s : vs_)
    for (int i = 0; i < 4; ++i) s.hist[i] = be_.alloc(DType::F32, {kMaxHist, 3, s.conv[i].dim(1)}, "conv.hist");
}

void Model::rollback(int64_t n_keep) {
  if (hist_rows_ == 0 || n_keep < 1 || n_keep > hist_rows_) throw std::runtime_error("rollback: no history");
  const int64_t base = len_ - hist_rows_;
  if (n_keep < hist_rows_) {
    for (auto& s : vs_)
      for (int i = 0; i < 4; ++i) {
        size_t row = s.conv[i].nbytes();
        be_.copy(s.hist[i], s.conv[i], size_t(n_keep - 1) * row, 0, row);
      }
    set_len(base + n_keep);
  }
  hist_rows_ = 0;
}

void Model::debug_visit(int v, const Tensor& x_in, const std::vector<const Tensor*>& blocks_in,
                        const std::vector<const Tensor*>& bank_in, Tensor& x_out, int64_t T) {
  const int64_t D = cfg_.hidden;
  const int L = cfg_.visits[v];
  write_tau(T);
  Tensor x = view(x_, {T, D});
  be_.copy(x_in, x);
  std::vector<Tensor> blocks;
  for (size_t i = 0; i < blocks_in.size(); ++i) {
    Tensor b = view(blk_[i], {T, D});
    be_.copy(*blocks_in[i], b);
    blocks.push_back(b);
  }
  ChainMsg cmsg;
  if (!bank_in.empty()) {
    for (size_t i = 0; i < bank_in.size(); ++i) {
      Tensor b = view(bank_[i], {T, D});
      be_.copy(*bank_in[i], b);
    }
    cmsg.bank0 = view(bank_[0], {T, D});
    cmsg.bank1 = view(bank_[1], {T, D});
    cmsg.queries = row_view(msg_queries_, L - cfg_.core_start, {4, D});
  }
  no_chain_ = true;  // the whole visit must finish here (no tail left for a next visit)
  run_visit(v, x, blocks, bank_in.empty() ? nullptr : &cmsg, T);
  no_chain_ = false;
  be_.copy(x, x_out);
  be_.sync();
}

}  // namespace ink
