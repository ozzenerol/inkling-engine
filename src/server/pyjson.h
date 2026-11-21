// Small order-preserving JSON value with a parser and a serializer that matches Python's json.dumps
// (ensure_ascii=False, float repr, separators, sort_keys). The chat template's `tojson` filter is
// json.dumps, so prompt bytes depend on this matching exactly. Owner: C.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ink::pyjson {

enum class Kind : uint8_t { Null, Bool, Int, Float, String, Array, Object };

struct Value;
using Array = std::vector<Value>;
using Member = std::pair<std::string, Value>;
using Object = std::vector<Member>;

struct Value {
  Kind kind = Kind::Null;
  bool b = false;
  double d = 0;
  std::string s;  // String: the text; Int: the decimal digits (arbitrary size, as Python keeps them)
  Array arr;
  Object obj;

  Value() = default;
  Value(std::nullptr_t) {}
  Value(bool v) : kind(Kind::Bool), b(v) {}
  Value(int v) : kind(Kind::Int), s(std::to_string(v)) {}
  Value(int64_t v) : kind(Kind::Int), s(std::to_string(v)) {}
  Value(uint64_t v) : kind(Kind::Int), s(std::to_string(v)) {}
  Value(double v) : kind(Kind::Float), d(v) {}
  Value(const char* v) : kind(Kind::String), s(v) {}
  Value(std::string v) : kind(Kind::String), s(std::move(v)) {}
  Value(std::string_view v) : kind(Kind::String), s(v) {}
  static Value array(Array a = {}) { Value v; v.kind = Kind::Array; v.arr = std::move(a); return v; }
  static Value object(Object o = {}) { Value v; v.kind = Kind::Object; v.obj = std::move(o); return v; }

  bool is_null() const { return kind == Kind::Null; }
  bool is_bool() const { return kind == Kind::Bool; }
  bool is_string() const { return kind == Kind::String; }
  bool is_number() const { return kind == Kind::Int || kind == Kind::Float; }
  bool is_array() const { return kind == Kind::Array; }
  bool is_object() const { return kind == Kind::Object; }

  // Python truthiness (what Jinja `if x` tests).
  bool truthy() const;

  // Object access. find() returns nullptr when missing. set() replaces in place or appends.
  const Value* find(std::string_view key) const;
  Value* find(std::string_view key);
  Value& set(std::string_view key, Value v);
  bool erase(std::string_view key);
  const Value& operator[](std::string_view key) const;  // null value when missing or not an object

  // Convenience getters with defaults.
  std::string str_or(std::string_view key, std::string_view def = "") const;
  double num_or(std::string_view key, double def) const;
  int64_t int_or(std::string_view key, int64_t def) const;
  bool bool_or(std::string_view key, bool def) const;
  double as_double() const;  // Int or Float
};

struct ParseError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Parses one JSON document (RFC 8259 plus NaN/Infinity/-Infinity like Python). Duplicate keys keep the
// first position and the last value, like a Python dict.
Value parse(std::string_view text);

struct DumpOptions {
  std::string_view item_sep = ", ";  // Python default when indent is None
  std::string_view key_sep = ": ";
  bool sort_keys = false;
  bool ensure_ascii = false;
};
std::string dump(const Value& v, const DumpOptions& o = {});
void dump_to(std::string& out, const Value& v, const DumpOptions& o = {});
// Compact form for wire output: separators (",", ":"), not sorted.
std::string compact(const Value& v);

// Python repr(float) / str(float).
std::string py_float(double d);
// Python str() of a scalar, as Jinja's `~` would print it (strings raw, None, True/False, numbers).
std::string py_str(const Value& v);
// JSON string literal with Python's escaping.
void quote_to(std::string& out, std::string_view s, bool ensure_ascii = false);

}  // namespace ink::pyjson
