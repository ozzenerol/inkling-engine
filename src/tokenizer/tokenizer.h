// Inkling 2 tokenizer: tokenizer.json byte-level BPE (o200k-style split regex, ignore_merges) plus the
// added special tokens. Matches HF `tokenizers` encode(add_special_tokens=False). Owner: C.
#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ink {

class Tokenizer {
 public:
  // dir holds tokenizer.json. Throws std::runtime_error on failure.
  static std::unique_ptr<Tokenizer> load(const std::string& dir);
  ~Tokenizer();

  // allow_special: text like "<|end_message|>" becomes the special token (what HF does with a rendered
  // chat template). Without it, special-token text is tokenized as plain text.
  std::vector<int32_t> encode(std::string_view text, bool allow_special = true) const;
  void encode_append(std::string_view text, std::vector<int32_t>& out, bool allow_special = true) const;

  std::string decode(std::span<const int32_t> ids) const;
  // Raw bytes of one token (special tokens: their text). Empty for unknown ids.
  std::string_view token_bytes(int32_t id) const;
  bool is_special(int32_t id) const;
  // id of a special token by its text, -1 if none.
  int32_t special(std::string_view text) const;
  int32_t vocab_size() const;  // max id + 1

 private:
  Tokenizer();
  struct Impl;
  std::unique_ptr<Impl> p_;
};

// Incremental detokenizer for streaming: returns only complete UTF-8, keeps a partial sequence pending.
class StreamDecoder {
 public:
  explicit StreamDecoder(const Tokenizer& t) : t_(t) {}
  std::string push(int32_t id);
  std::string push_bytes(std::string_view bytes);
  std::string flush();  // whatever is pending (invalid tails are emitted as U+FFFD)

 private:
  const Tokenizer& t_;
  std::string pending_;
};

}  // namespace ink
