#include "core/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace ink::json {

namespace {
const Value kNull;

struct Parser {
  std::string_view s;
  size_t i = 0;

  [[noreturn]] void fail(const char* what) {
    throw std::runtime_error(std::string("json: ") + what + " at offset " + std::to_string(i));
  }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
  }
  bool lit(const char* t) {
    size_t n = std::char_traits<char>::length(t);
    if (s.substr(i, n) == t) { i += n; return true; }
    return false;
  }
  static void utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += char(cp);
    else if (cp < 0x800) { o += char(0xC0 | (cp >> 6)); o += char(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += char(0xE0 | (cp >> 12)); o += char(0x80 | ((cp >> 6) & 0x3F)); o += char(0x80 | (cp & 0x3F)); }
    else { o += char(0xF0 | (cp >> 18)); o += char(0x80 | ((cp >> 12) & 0x3F)); o += char(0x80 | ((cp >> 6) & 0x3F)); o += char(0x80 | (cp & 0x3F)); }
  }
  uint32_t hex4() {
    if (i + 4 > s.size()) fail("bad \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad hex");
    }
    return v;
  }
  std::string string() {
    if (s[i] != '"') fail("expected string");
    ++i;
    std::string o;
    while (true) {
      if (i >= s.size()) fail("unterminated string");
      char c = s[i++];
      if (c == '"') break;
      if (c != '\\') { o += c; continue; }
      if (i >= s.size()) fail("bad escape");
      char e = s[i++];
      switch (e) {
        case '"': o += '"'; break;
        case '\\': o += '\\'; break;
        case '/': o += '/'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'n': o += '\n'; break;
        case 'r': o += '\r'; break;
        case 't': o += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
            i += 2;
            uint32_t lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          utf8(o, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
    return o;
  }
  Value value() {
    ws();
    if (i >= s.size()) fail("unexpected end");
    Value v;
    char c = s[i];
    if (c == '{') {
      v.kind = Value::Object;
      ++i; ws();
      if (s[i] == '}') { ++i; return v; }
      while (true) {
        ws();
        std::string k = string();
        ws();
        if (s[i] != ':') fail("expected :");
        ++i;
        v.obj[k] = value();
        ws();
        if (s[i] == ',') { ++i; continue; }
        if (s[i] == '}') { ++i; break; }
        fail("expected , or }");
      }
    } else if (c == '[') {
      v.kind = Value::Array;
      ++i; ws();
      if (s[i] == ']') { ++i; return v; }
      while (true) {
        v.arr.push_back(value());
        ws();
        if (s[i] == ',') { ++i; continue; }
        if (s[i] == ']') { ++i; break; }
        fail("expected , or ]");
      }
    } else if (c == '"') {
      v.kind = Value::String;
      v.str = string();
    } else if (lit("true")) { v.kind = Value::Bool; v.b = true; }
    else if (lit("false")) { v.kind = Value::Bool; v.b = false; }
    else if (lit("null")) { v.kind = Value::Null; }
    else {
      const char* begin = s.data() + i;
      char* end = nullptr;
      std::string tmp(begin, std::min<size_t>(64, s.size() - i));
      double d = std::strtod(tmp.c_str(), &end);
      if (end == tmp.c_str()) fail("bad value");
      i += size_t(end - tmp.c_str());
      v.kind = Value::Number;
      v.num = d;
    }
    return v;
  }
};

void dump_to(std::string& o, const Value& v) {
  switch (v.kind) {
    case Value::Null: o += "null"; break;
    case Value::Bool: o += v.b ? "true" : "false"; break;
    case Value::Number: {
      char buf[32];
      if (std::isfinite(v.num) && v.num == double(int64_t(v.num)) && std::fabs(v.num) < 1e15)
        std::snprintf(buf, sizeof buf, "%lld", (long long)v.num);
      else
        std::snprintf(buf, sizeof buf, "%.17g", v.num);
      o += buf;
      break;
    }
    case Value::String: o += '"'; o += escape(v.str); o += '"'; break;
    case Value::Array:
      o += '[';
      for (size_t k = 0; k < v.arr.size(); ++k) { if (k) o += ','; dump_to(o, v.arr[k]); }
      o += ']';
      break;
    case Value::Object: {
      o += '{';
      bool first = true;
      for (auto& [k, x] : v.obj) {
        if (!first) o += ',';
        first = false;
        o += '"'; o += escape(k); o += "\":";
        dump_to(o, x);
      }
      o += '}';
      break;
    }
  }
}
}  // namespace

const Value& Value::operator[](const std::string& k) const {
  if (kind != Object) throw std::runtime_error("json: not an object (key " + k + ")");
  auto it = obj.find(k);
  return it == obj.end() ? kNull : it->second;
}
double Value::as_num() const {
  if (kind != Number) throw std::runtime_error("json: not a number");
  return num;
}
const std::string& Value::as_str() const {
  if (kind != String) throw std::runtime_error("json: not a string");
  return str;
}
bool Value::as_bool() const {
  if (kind != Bool) throw std::runtime_error("json: not a bool");
  return b;
}

Value parse(std::string_view text) {
  Parser p{text};
  Value v = p.value();
  p.ws();
  if (p.i != text.size()) p.fail("trailing data");
  return v;
}

std::string dump(const Value& v) {
  std::string o;
  dump_to(o, v);
  return o;
}

std::string escape(std::string_view s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          o += buf;
        } else {
          o += char(c);
        }
    }
  }
  return o;
}

}  // namespace ink::json
