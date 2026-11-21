#include "json.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace ink::json {

static const Value kNull;

bool Value::truthy() const {
  switch (kind) {
    case Kind::Null: return false;
    case Kind::Bool: return b;
    case Kind::Int: return !(s == "0" || s == "-0");
    case Kind::Float: return d != 0.0;  // NaN is truthy in Python
    case Kind::String: return !s.empty();
    case Kind::Array: return !arr.empty();
    case Kind::Object: return !obj.empty();
  }
  return false;
}

const Value* Value::find(std::string_view key) const {
  if (kind != Kind::Object) return nullptr;
  for (auto& m : obj)
    if (m.first == key) return &m.second;
  return nullptr;
}
Value* Value::find(std::string_view key) {
  return const_cast<Value*>(static_cast<const Value*>(this)->find(key));
}
Value& Value::set(std::string_view key, Value v) {
  if (kind != Kind::Object) { *this = object(); }
  for (auto& m : obj)
    if (m.first == key) { m.second = std::move(v); return m.second; }
  obj.emplace_back(std::string(key), std::move(v));
  return obj.back().second;
}
bool Value::erase(std::string_view key) {
  if (kind != Kind::Object) return false;
  for (auto it = obj.begin(); it != obj.end(); ++it)
    if (it->first == key) { obj.erase(it); return true; }
  return false;
}
const Value& Value::operator[](std::string_view key) const {
  const Value* v = find(key);
  return v ? *v : kNull;
}
std::string Value::str_or(std::string_view key, std::string_view def) const {
  const Value* v = find(key);
  return v && v->is_string() ? v->s : std::string(def);
}
double Value::as_double() const {
  if (kind == Kind::Float) return d;
  if (kind == Kind::Int) return std::strtod(s.c_str(), nullptr);
  if (kind == Kind::Bool) return b ? 1 : 0;
  return 0;
}
double Value::num_or(std::string_view key, double def) const {
  const Value* v = find(key);
  return v && v->is_number() ? v->as_double() : def;
}
int64_t Value::int_or(std::string_view key, int64_t def) const {
  const Value* v = find(key);
  if (!v || !v->is_number()) return def;
  if (v->kind == Kind::Int) return std::strtoll(v->s.c_str(), nullptr, 10);
  return (int64_t)v->d;
}
bool Value::bool_or(std::string_view key, bool def) const {
  const Value* v = find(key);
  return v && v->is_bool() ? v->b : def;
}

// ---------------------------------------------------------------- parser

namespace {

struct Parser {
  std::string_view t;
  size_t i = 0;
  int depth = 0;

  [[noreturn]] void fail(const char* what) {
    throw ParseError(std::string("JSON: ") + what + " at offset " + std::to_string(i));
  }
  void ws() {
    while (i < t.size() && (t[i] == ' ' || t[i] == '\t' || t[i] == '\n' || t[i] == '\r')) i++;
  }
  bool lit(std::string_view w) {
    if (t.substr(i, w.size()) == w) { i += w.size(); return true; }
    return false;
  }
  static void put_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
      out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
    } else {
      out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
      out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
    }
  }
  uint32_t hex4() {
    if (i + 4 > t.size()) fail("short \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; k++) {
      char c = t[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad \\u escape");
    }
    return v;
  }
  std::string str() {
    // at opening quote
    i++;
    std::string out;
    for (;;) {
      size_t j = i;
      while (j < t.size() && t[j] != '"' && t[j] != '\\' && (unsigned char)t[j] >= 0x20) j++;
      out.append(t.data() + i, j - i);
      i = j;
      if (i >= t.size()) fail("unterminated string");
      char c = t[i];
      if (c == '"') { i++; return out; }
      if ((unsigned char)c < 0x20) fail("control character in string");
      i++;  // backslash
      if (i >= t.size()) fail("bad escape");
      char e = t[i++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < t.size() && t[i] == '\\' && t[i + 1] == 'u') {
            size_t save = i;
            i += 2;
            uint32_t lo = hex4();
            if (lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            else i = save;  // lone high surrogate; keep it as is
          }
          put_utf8(out, cp);  // lone surrogates become (invalid) 3-byte sequences, as CESU would
          break;
        }
        default: fail("bad escape");
      }
    }
  }
  Value number() {
    size_t st = i;
    if (t[i] == '-') i++;
    if (i < t.size() && lit("Infinity")) { Value v(-INFINITY); return v; }
    if (i >= t.size() || !(t[i] >= '0' && t[i] <= '9')) fail("bad number");
    if (t[i] == '0') i++;
    else while (i < t.size() && t[i] >= '0' && t[i] <= '9') i++;
    bool is_float = false;
    if (i < t.size() && t[i] == '.') {
      is_float = true; i++;
      if (i >= t.size() || !(t[i] >= '0' && t[i] <= '9')) fail("bad fraction");
      while (i < t.size() && t[i] >= '0' && t[i] <= '9') i++;
    }
    if (i < t.size() && (t[i] == 'e' || t[i] == 'E')) {
      is_float = true; i++;
      if (i < t.size() && (t[i] == '+' || t[i] == '-')) i++;
      if (i >= t.size() || !(t[i] >= '0' && t[i] <= '9')) fail("bad exponent");
      while (i < t.size() && t[i] >= '0' && t[i] <= '9') i++;
    }
    std::string_view txt = t.substr(st, i - st);
    Value v;
    if (is_float) {
      v.kind = Kind::Float;
      v.d = std::strtod(std::string(txt).c_str(), nullptr);
    } else {
      v.kind = Kind::Int;
      v.s = txt == "-0" ? "0" : std::string(txt);
    }
    return v;
  }
  Value value() {
    ws();
    if (i >= t.size()) fail("unexpected end");
    char c = t[i];
    if (c == '"') return Value(str());
    if (c == '{') {
      if (++depth > 512) fail("too deep");
      i++;
      Value v = Value::object();
      ws();
      if (i < t.size() && t[i] == '}') { i++; depth--; return v; }
      std::unordered_map<std::string, size_t> index;  // built once the object gets large
      for (;;) {
        ws();
        if (i >= t.size() || t[i] != '"') fail("expected key");
        std::string k = str();
        ws();
        if (i >= t.size() || t[i] != ':') fail("expected ':'");
        i++;
        Value x = value();
        if (v.obj.size() < 32) {
          v.set(k, std::move(x));
        } else {
          if (index.empty())
            for (size_t n = 0; n < v.obj.size(); n++) index.emplace(v.obj[n].first, n);
          auto [it, fresh] = index.emplace(k, v.obj.size());
          if (fresh) v.obj.emplace_back(std::move(k), std::move(x));
          else v.obj[it->second].second = std::move(x);
        }
        ws();
        if (i < t.size() && t[i] == ',') { i++; continue; }
        if (i < t.size() && t[i] == '}') { i++; break; }
        fail("expected ',' or '}'");
      }
      depth--;
      return v;
    }
    if (c == '[') {
      if (++depth > 512) fail("too deep");
      i++;
      Value v = Value::array();
      ws();
      if (i < t.size() && t[i] == ']') { i++; depth--; return v; }
      for (;;) {
        v.arr.push_back(value());
        ws();
        if (i < t.size() && t[i] == ',') { i++; continue; }
        if (i < t.size() && t[i] == ']') { i++; break; }
        fail("expected ',' or ']'");
      }
      depth--;
      return v;
    }
    if (lit("true")) return Value(true);
    if (lit("false")) return Value(false);
    if (lit("null")) return Value();
    if (lit("NaN")) return Value(NAN);
    if (lit("Infinity")) return Value(INFINITY);
    if (c == '-' || (c >= '0' && c <= '9')) return number();
    fail("unexpected character");
  }
};

}  // namespace

Value parse(std::string_view text) {
  Parser p{text};
  Value v = p.value();
  p.ws();
  if (p.i != text.size()) p.fail("trailing data");
  return v;
}

// ---------------------------------------------------------------- serializer

std::string py_float(double d) {
  if (std::isnan(d)) return "NaN";  // json.dumps spelling; repr would be 'nan'
  if (std::isinf(d)) return d > 0 ? "Infinity" : "-Infinity";
  if (d == 0) return std::signbit(d) ? "-0.0" : "0.0";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
  std::string sci(buf, r.ptr);  // shortest round-trip, e.g. "-1.2345e+02"
  bool neg = sci[0] == '-';
  if (neg) sci.erase(0, 1);
  size_t epos = sci.find('e');
  int exp = std::atoi(sci.c_str() + epos + 1);
  std::string digits;
  for (size_t k = 0; k < epos; k++)
    if (sci[k] != '.') digits += sci[k];
  std::string out = neg ? "-" : "";
  if (exp < -4 || exp >= 16) {
    out += digits[0];
    if (digits.size() > 1) { out += '.'; out.append(digits, 1); }
    char eb[16];
    std::snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', exp < 0 ? -exp : exp);
    out += eb;
  } else if (exp < 0) {
    out += "0.";
    out.append(-exp - 1, '0');
    out += digits;
  } else {
    if ((int)digits.size() <= exp + 1) {
      out += digits;
      out.append(exp + 1 - digits.size(), '0');
      out += ".0";
    } else {
      out.append(digits, 0, exp + 1);
      out += '.';
      out.append(digits, exp + 1);
    }
  }
  return out;
}

void quote_to(std::string& out, std::string_view s, bool ensure_ascii) {
  static const char* hex = "0123456789abcdef";
  out += '"';
  for (size_t k = 0; k < s.size(); k++) {
    unsigned char c = (unsigned char)s[k];
    switch (c) {
      case '"': out += "\\\""; continue;
      case '\\': out += "\\\\"; continue;
      case '\n': out += "\\n"; continue;
      case '\r': out += "\\r"; continue;
      case '\t': out += "\\t"; continue;
      case '\b': out += "\\b"; continue;
      case '\f': out += "\\f"; continue;
    }
    if (c < 0x20) {
      out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15];
    } else if (c >= 0x80 && ensure_ascii) {
      // decode one UTF-8 sequence
      uint32_t cp = 0; int n = 0;
      if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; n = 1; }
      else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; n = 2; }
      else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; n = 3; }
      for (int j = 0; j < n && k + 1 < s.size(); j++) cp = (cp << 6) | ((unsigned char)s[++k] & 0x3F);
      auto u = [&](uint32_t x) {
        out += "\\u"; out += hex[(x >> 12) & 15]; out += hex[(x >> 8) & 15]; out += hex[(x >> 4) & 15]; out += hex[x & 15];
      };
      if (cp >= 0x10000) { cp -= 0x10000; u(0xD800 + (cp >> 10)); u(0xDC00 + (cp & 0x3FF)); }
      else u(cp);
    } else {
      out += (char)c;
    }
  }
  out += '"';
}

void dump_to(std::string& out, const Value& v, const DumpOptions& o) {
  switch (v.kind) {
    case Kind::Null: out += "null"; return;
    case Kind::Bool: out += v.b ? "true" : "false"; return;
    case Kind::Int: out += v.s; return;
    case Kind::Float: out += py_float(v.d); return;
    case Kind::String: quote_to(out, v.s, o.ensure_ascii); return;
    case Kind::Array:
      out += '[';
      for (size_t k = 0; k < v.arr.size(); k++) {
        if (k) out += o.item_sep;
        dump_to(out, v.arr[k], o);
      }
      out += ']';
      return;
    case Kind::Object: {
      out += '{';
      std::vector<const Member*> ms;
      ms.reserve(v.obj.size());
      for (auto& m : v.obj) ms.push_back(&m);
      // Python compares str by code point; UTF-8 byte order is the same order.
      if (o.sort_keys)
        std::stable_sort(ms.begin(), ms.end(), [](const Member* a, const Member* b) { return a->first < b->first; });
      for (size_t k = 0; k < ms.size(); k++) {
        if (k) out += o.item_sep;
        quote_to(out, ms[k]->first, o.ensure_ascii);
        out += o.key_sep;
        dump_to(out, ms[k]->second, o);
      }
      out += '}';
      return;
    }
  }
}

std::string dump(const Value& v, const DumpOptions& o) {
  std::string s;
  dump_to(s, v, o);
  return s;
}

std::string compact(const Value& v) { return dump(v, DumpOptions{",", ":", false, false}); }

std::string py_str(const Value& v) {
  switch (v.kind) {
    case Kind::Null: return "None";
    case Kind::Bool: return v.b ? "True" : "False";
    case Kind::Int: return v.s;
    case Kind::Float:
      if (std::isnan(v.d)) return "nan";
      if (std::isinf(v.d)) return v.d > 0 ? "inf" : "-inf";
      return py_float(v.d);
    case Kind::String: return v.s;
    default: return dump(v);  // not exact Python repr for containers; templates never print those
  }
}

}  // namespace ink::json
