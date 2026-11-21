#include "core/safetensors.h"

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <stdexcept>

#include "core/json.h"

namespace ink {

const char* dtype_name(DType d) {
  switch (d) {
    case DType::F32: return "f32";
    case DType::BF16: return "bf16";
    case DType::F16: return "f16";
    case DType::U8: return "u8";
    case DType::U32: return "u32";
    case DType::I32: return "i32";
  }
  return "?";
}

bool dtype_from_safetensors(const std::string& s, DType* out) {
  if (s == "F32") *out = DType::F32;
  else if (s == "BF16") *out = DType::BF16;
  else if (s == "F16") *out = DType::F16;
  else if (s == "U8") *out = DType::U8;
  else if (s == "U32") *out = DType::U32;
  else if (s == "I32") *out = DType::I32;
  else return false;
  return true;
}

std::string Tensor::str() const {
  std::string s = dtype_name(dtype);
  s += "[";
  for (int i = 0; i < ndim; ++i) s += (i ? "," : "") + std::to_string(shape[i]);
  return s + "]";
}

void SafeTensors::add_file(const std::string& path, const std::string& prefix) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  uint64_t n = 0;
  if (pread(fd, &n, 8, 0) != 8 || n > (1u << 30)) {
    close(fd);
    throw std::runtime_error("bad safetensors header " + path);
  }
  std::string hdr(n, '\0');
  if (pread(fd, hdr.data(), n, 8) != ssize_t(n)) {
    close(fd);
    throw std::runtime_error("short header " + path);
  }
  close(fd);
  json::Value h = json::parse(hdr);
  uint64_t base = 8 + n;
  for (auto& [k, v] : h.obj) {
    if (k == "__metadata__") continue;
    Entry e;
    e.name = prefix + k;
    e.file = path;
    if (!dtype_from_safetensors(v["dtype"].as_str(), &e.dtype))
      throw std::runtime_error("unsupported dtype " + v["dtype"].as_str() + " for " + k);
    auto& sh = v["shape"].arr;
    if (sh.size() > 4) throw std::runtime_error("rank > 4: " + k);
    e.ndim = int(sh.size());
    for (size_t i = 0; i < sh.size(); ++i) e.shape[i] = sh[i].as_int();
    uint64_t a = uint64_t(v["data_offsets"][0].as_int()), b = uint64_t(v["data_offsets"][1].as_int());
    e.offset = base + a;
    e.bytes = b - a;
    entries_[e.name] = e;
  }
}

void SafeTensors::add_dir(const std::string& dir, const std::string& exclude_substr) {
  DIR* d = opendir(dir.c_str());
  if (!d) throw std::runtime_error("cannot open dir " + dir);
  std::vector<std::string> files;
  while (dirent* de = readdir(d)) {
    std::string f = de->d_name;
    if (f.size() > 12 && f.substr(f.size() - 12) == ".safetensors" &&
        (exclude_substr.empty() || f.find(exclude_substr) == std::string::npos))
      files.push_back(dir + "/" + f);
  }
  closedir(d);
  std::sort(files.begin(), files.end());
  for (auto& f : files) add_file(f);
}

const SafeTensors::Entry* SafeTensors::find(const std::string& name) const {
  auto it = entries_.find(name);
  return it == entries_.end() ? nullptr : &it->second;
}

const SafeTensors::Entry& SafeTensors::at(const std::string& name) const {
  auto* e = find(name);
  if (!e) throw std::runtime_error("missing tensor " + name);
  return *e;
}

uint64_t SafeTensors::total_bytes() const {
  uint64_t t = 0;
  for (auto& [k, e] : entries_) t += e.bytes;
  return t;
}

void read_entry(const SafeTensors::Entry& e, void* dst) {
  int fd = open(e.file.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + e.file);
  uint64_t done = 0;
  while (done < e.bytes) {
    size_t chunk = size_t(std::min<uint64_t>(e.bytes - done, 1ull << 30));
    ssize_t r = pread(fd, static_cast<char*>(dst) + done, chunk, off_t(e.offset + done));
    if (r <= 0) {
      close(fd);
      throw std::runtime_error("read failed " + e.name);
    }
    done += uint64_t(r);
  }
  close(fd);
}

}  // namespace ink
