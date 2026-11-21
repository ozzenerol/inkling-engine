// Owner: A. Minimal JSON value + parser (enough for config.json and safetensors headers). Others may use it.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ink::json {

struct Value {
  enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Value> arr;
  std::map<std::string, Value> obj;

  bool has(const std::string& k) const { return kind == Object && obj.count(k); }
  const Value& operator[](const std::string& k) const;
  const Value& operator[](size_t i) const { return arr.at(i); }
  double as_num() const;
  int64_t as_int() const { return int64_t(as_num()); }
  const std::string& as_str() const;
  bool as_bool() const;
  size_t size() const { return kind == Array ? arr.size() : obj.size(); }
  // Convenience with defaults.
  double get_num(const std::string& k, double def) const { return has(k) && obj.at(k).kind == Number ? obj.at(k).num : def; }
  int64_t get_int(const std::string& k, int64_t def) const { return int64_t(get_num(k, double(def))); }
};

Value parse(std::string_view text);           // throws std::runtime_error
std::string dump(const Value& v);             // compact
std::string escape(std::string_view s);       // JSON string body escaping (no quotes)

}  // namespace ink::json
