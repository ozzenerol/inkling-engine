#include "tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <fstream>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "../server/pyjson.h"

namespace ink {

namespace {

// GPT-2 byte <-> unicode map used by the ByteLevel pre-tokenizer.
struct ByteMap {
  uint32_t byte_to_cp[256];
  std::unordered_map<uint32_t, uint8_t> cp_to_byte;
  ByteMap() {
    int n = 0;
    for (int b = 0; b < 256; b++) {
      bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
      byte_to_cp[b] = keep ? b : 256 + n++;
      cp_to_byte[byte_to_cp[b]] = (uint8_t)b;
    }
  }
};

// Decode a byte-level-encoded vocab string (UTF-8 of mapped code points) to raw bytes.
bool bytelevel_to_raw(const ByteMap& bm, std::string_view s, std::string& out) {
  out.clear();
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    uint32_t cp;
    int n;
    if (c < 0x80) { cp = c; n = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; n = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; n = 3; }
    else { cp = c & 0x07; n = 4; }
    if (i + n > s.size()) return false;
    for (int k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    i += n;
    auto it = bm.cp_to_byte.find(cp);
    if (it == bm.cp_to_byte.end()) return false;
    out += (char)it->second;
  }
  return true;
}

// Length of a valid UTF-8 sequence at s[i], or 0 if invalid.
size_t utf8_len(std::string_view s, size_t i) {
  unsigned char c = s[i];
  size_t n;
  uint32_t cp;
  if (c < 0x80) return 1;
  if ((c & 0xE0) == 0xC0) { n = 2; cp = c & 0x1F; }
  else if ((c & 0xF0) == 0xE0) { n = 3; cp = c & 0x0F; }
  else if ((c & 0xF8) == 0xF0) { n = 4; cp = c & 0x07; }
  else return 0;
  if (i + n > s.size()) return 0;
  for (size_t k = 1; k < n; k++) {
    unsigned char d = s[i + k];
    if ((d & 0xC0) != 0x80) return 0;
    cp = (cp << 6) | (d & 0x3F);
  }
  if ((n == 2 && cp < 0x80) || (n == 3 && cp < 0x800) || (n == 4 && (cp < 0x10000 || cp > 0x10FFFF)) ||
      (cp >= 0xD800 && cp < 0xE000))
    return 0;
  return n;
}

std::string sanitize_utf8(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    size_t n = utf8_len(s, i);
    if (n == 0) { out += "\xEF\xBF\xBD"; i++; }
    else { out.append(s.data() + i, n); i += n; }
  }
  return out;
}

bool valid_utf8(std::string_view s) {
  for (size_t i = 0; i < s.size();) {
    if ((unsigned char)s[i] < 0x80) { i++; continue; }
    size_t n = utf8_len(s, i);
    if (!n) return false;
    i += n;
  }
  return true;
}

struct PairHash {
  size_t operator()(uint64_t k) const { return std::hash<uint64_t>()(k * 0x9E3779B97F4A7C15ull); }
};

}  // namespace

struct Tokenizer::Impl {
  std::vector<std::string> id_bytes;       // raw bytes per id
  std::vector<uint8_t> special;            // per id
  std::unordered_map<std::string, int32_t> vocab;  // raw bytes -> id (BPE vocab only)
  std::unordered_map<uint64_t, std::pair<int32_t, int32_t>, PairHash> merges;  // (l<<32|r) -> (rank, id)
  int32_t byte_id[256];
  std::vector<std::pair<std::string, int32_t>> specials;  // all added tokens, text -> id
  std::unordered_map<std::string, int32_t> special_by_text;
  pcre2_code* re = nullptr;
  bool jit = false;
  // BPE cache for pieces not in the vocab
  mutable std::mutex mu;
  mutable std::unordered_map<std::string, std::vector<int32_t>> cache;

  ~Impl() {
    if (re) pcre2_code_free(re);
  }

  void bpe(std::string_view piece, std::vector<int32_t>& out) const {
    auto it = vocab.find(std::string(piece));
    if (it != vocab.end()) { out.push_back(it->second); return; }  // ignore_merges
    {
      std::lock_guard<std::mutex> g(mu);
      auto c = cache.find(std::string(piece));
      if (c != cache.end()) { out.insert(out.end(), c->second.begin(), c->second.end()); return; }
    }
    const int n = (int)piece.size();
    std::vector<int32_t> sym(n), prev(n), next(n);
    std::vector<uint8_t> alive(n, 1);
    for (int i = 0; i < n; i++) {
      sym[i] = byte_id[(unsigned char)piece[i]];
      prev[i] = i - 1;
      next[i] = i + 1 < n ? i + 1 : -1;
    }
    // min-heap of (rank, left position, left sym, right sym); stale entries skipped
    struct Cand { int32_t rank, pos, l, r; };
    auto cmp = [](const Cand& a, const Cand& b) { return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos; };
    std::priority_queue<Cand, std::vector<Cand>, decltype(cmp)> pq(cmp);
    auto push = [&](int i) {
      if (i < 0) return;
      int j = next[i];
      if (j < 0) return;
      auto m = merges.find(((uint64_t)(uint32_t)sym[i] << 32) | (uint32_t)sym[j]);
      if (m != merges.end()) pq.push({m->second.first, i, sym[i], sym[j]});
    };
    for (int i = 0; i + 1 < n; i++) push(i);
    while (!pq.empty()) {
      Cand c = pq.top();
      pq.pop();
      int i = c.pos;
      if (!alive[i] || sym[i] != c.l) continue;
      int j = next[i];
      if (j < 0 || sym[j] != c.r) continue;
      sym[i] = merges.at(((uint64_t)(uint32_t)c.l << 32) | (uint32_t)c.r).second;
      alive[j] = 0;
      next[i] = next[j];
      if (next[j] >= 0) prev[next[j]] = i;
      push(prev[i]);
      push(i);
    }
    std::vector<int32_t> res;
    for (int i = 0; i >= 0; i = next[i]) res.push_back(sym[i]);
    out.insert(out.end(), res.begin(), res.end());
    std::lock_guard<std::mutex> g(mu);
    if (cache.size() > 200000) cache.clear();
    cache.emplace(std::string(piece), std::move(res));
  }

  void encode_plain(std::string_view text, std::vector<int32_t>& out) const {
    if (text.empty()) return;
    pcre2_match_data* md = pcre2_match_data_create_from_pattern(re, nullptr);
    size_t off = 0;
    const auto* subj = reinterpret_cast<PCRE2_SPTR>(text.data());
    while (off < text.size()) {
      int rc = jit ? pcre2_jit_match(re, subj, text.size(), off, 0, md, nullptr)
                   : pcre2_match(re, subj, text.size(), off, 0, md, nullptr);
      if (rc < 0) {  // no more matches: the rest is one gap piece
        bpe(text.substr(off), out);
        break;
      }
      PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
      size_t s = ov[0], e = ov[1];
      if (s > off) bpe(text.substr(off, s - off), out);  // Split "Isolated": gaps are pieces too
      if (e == s) {  // empty match (not expected with this pattern): step one code point
        size_t n = utf8_len(text, s);
        e = s + (n ? n : 1);
      }
      bpe(text.substr(s, e - s), out);
      off = e;
    }
    pcre2_match_data_free(md);
  }
};

Tokenizer::Tokenizer() : p_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;

std::unique_ptr<Tokenizer> Tokenizer::load(const std::string& dir) {
  std::ifstream f(dir + "/tokenizer.json", std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + dir + "/tokenizer.json");
  std::stringstream ss;
  ss << f.rdbuf();
  pyjson::Value tj = pyjson::parse(ss.str());

  std::unique_ptr<Tokenizer> t(new Tokenizer());
  Impl& p = *t->p_;
  static const ByteMap bm;

  const pyjson::Value& model = tj["model"];
  if (model.str_or("type") != "BPE") throw std::runtime_error("tokenizer: model is not BPE");
  if (!model["ignore_merges"].truthy()) throw std::runtime_error("tokenizer: expected ignore_merges=true");
  int32_t max_id = -1;
  std::string raw;
  for (auto& [k, v] : model["vocab"].obj) {
    int32_t id = (int32_t)std::stoll(v.s);
    if (!bytelevel_to_raw(bm, k, raw)) throw std::runtime_error("tokenizer: bad vocab entry");
    p.vocab.emplace(raw, id);
    if (id >= (int32_t)p.id_bytes.size()) p.id_bytes.resize(id + 1);
    p.id_bytes[id] = raw;
    max_id = std::max(max_id, id);
  }
  for (int b = 0; b < 256; b++) {
    auto it = p.vocab.find(std::string(1, (char)b));
    if (it == p.vocab.end()) throw std::runtime_error("tokenizer: byte missing from vocab");
    p.byte_id[b] = it->second;
  }
  int32_t rank = 0;
  std::string a, b;
  for (auto& m : model["merges"].arr) {
    std::string sa, sb;
    if (m.is_array()) { sa = m.arr[0].s; sb = m.arr[1].s; }
    else { auto sp = m.s.find(' '); sa = m.s.substr(0, sp); sb = m.s.substr(sp + 1); }
    if (!bytelevel_to_raw(bm, sa, a) || !bytelevel_to_raw(bm, sb, b)) throw std::runtime_error("tokenizer: bad merge");
    auto ia = p.vocab.find(a), ib = p.vocab.find(b), ic = p.vocab.find(a + b);
    if (ia == p.vocab.end() || ib == p.vocab.end() || ic == p.vocab.end())
      throw std::runtime_error("tokenizer: merge refers to unknown token");
    uint64_t key = ((uint64_t)(uint32_t)ia->second << 32) | (uint32_t)ib->second;
    p.merges.emplace(key, std::make_pair(rank, ic->second));  // first (lowest) rank wins
    rank++;
  }
  for (auto& at : tj["added_tokens"].arr) {
    int32_t id = (int32_t)at.int_or("id", -1);
    std::string content = at.str_or("content");
    if (id < 0) continue;
    if (id >= (int32_t)p.id_bytes.size()) p.id_bytes.resize(id + 1);
    p.id_bytes[id] = content;
    max_id = std::max(max_id, id);
    p.specials.emplace_back(content, id);
    p.special_by_text[content] = id;
  }
  p.special.assign(p.id_bytes.size(), 0);
  for (auto& [s, id] : p.specials) p.special[id] = 1;

  // pre-tokenizer regex
  const pyjson::Value& pre = tj["pre_tokenizer"];
  std::string pattern;
  for (auto& pt : pre["pretokenizers"].arr)
    if (pt.str_or("type") == "Split") pattern = pt["pattern"].str_or("Regex");
  if (pattern.empty()) throw std::runtime_error("tokenizer: no Split regex in pre_tokenizer");
  int err;
  PCRE2_SIZE eoff;
  p.re = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(), PCRE2_UTF | PCRE2_UCP, &err,
                       &eoff, nullptr);
  if (!p.re) {
    PCRE2_UCHAR msg[256];
    pcre2_get_error_message(err, msg, sizeof msg);
    throw std::runtime_error(std::string("tokenizer: regex: ") + (const char*)msg);
  }
  p.jit = pcre2_jit_compile(p.re, PCRE2_JIT_COMPLETE) == 0;
  return t;
}

void Tokenizer::encode_append(std::string_view text_in, std::vector<int32_t>& out, bool allow_special) const {
  std::string fixed;
  std::string_view text = text_in;
  if (!valid_utf8(text)) { fixed = sanitize_utf8(text); text = fixed; }
  if (!allow_special) { p_->encode_plain(text, out); return; }
  // Leftmost-longest match of added tokens (all are "<|...|>" here, but don't rely on that).
  size_t start = 0, i = 0;
  while (i < text.size()) {
    int32_t best = -1;
    size_t best_len = 0;
    if (text[i] == '<') {
      for (auto& [s, id] : p_->specials)
        if (s.size() > best_len && text.compare(i, s.size(), s) == 0) { best = id; best_len = s.size(); }
    } else {
      for (auto& [s, id] : p_->specials)
        if (!s.empty() && s[0] == text[i] && s.size() > best_len && text.compare(i, s.size(), s) == 0) {
          best = id; best_len = s.size();
        }
    }
    if (best >= 0) {
      p_->encode_plain(text.substr(start, i - start), out);
      out.push_back(best);
      i += best_len;
      start = i;
    } else {
      i++;
    }
  }
  p_->encode_plain(text.substr(start), out);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool allow_special) const {
  std::vector<int32_t> out;
  encode_append(text, out, allow_special);
  return out;
}

std::string Tokenizer::decode(std::span<const int32_t> ids) const {
  std::string s;
  for (int32_t id : ids) s += token_bytes(id);
  return s;
}

std::string_view Tokenizer::token_bytes(int32_t id) const {
  if (id < 0 || id >= (int32_t)p_->id_bytes.size()) return {};
  return p_->id_bytes[id];
}
bool Tokenizer::is_special(int32_t id) const {
  return id >= 0 && id < (int32_t)p_->special.size() && p_->special[id];
}
int32_t Tokenizer::special(std::string_view text) const {
  auto it = p_->special_by_text.find(std::string(text));
  return it == p_->special_by_text.end() ? -1 : it->second;
}
int32_t Tokenizer::vocab_size() const { return (int32_t)p_->id_bytes.size(); }

// ---------------------------------------------------------------- streaming

std::string StreamDecoder::push(int32_t id) { return push_bytes(t_.token_bytes(id)); }

std::string StreamDecoder::push_bytes(std::string_view bytes) {
  pending_.append(bytes);
  // emit the longest prefix that does not end inside a (possibly valid) UTF-8 sequence
  size_t i = 0, cut = 0;
  while (i < pending_.size()) {
    unsigned char c = pending_[i];
    size_t need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
    if (i + need > pending_.size()) {
      // incomplete tail: keep it if the bytes so far are continuation bytes
      bool ok = true;
      for (size_t k = i + 1; k < pending_.size(); k++) ok &= ((unsigned char)pending_[k] & 0xC0) == 0x80;
      if (ok) break;
    }
    i += need;
    cut = std::min(i, pending_.size());
  }
  std::string out = sanitize_utf8(std::string_view(pending_).substr(0, cut));
  pending_.erase(0, cut);
  return out;
}

std::string StreamDecoder::flush() {
  std::string out = sanitize_utf8(pending_);
  pending_.clear();
  return out;
}

}  // namespace ink
