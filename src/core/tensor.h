// Owner: A. Non-owning tensor view shared by all modules.
#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

namespace ink {

enum class DType : uint8_t { F32, BF16, F16, U8, U32, I32 };

inline size_t dtype_size(DType d) {
  switch (d) {
    case DType::F32: case DType::U32: case DType::I32: return 4;
    case DType::BF16: case DType::F16: return 2;
    case DType::U8: return 1;
  }
  return 0;
}
const char* dtype_name(DType d);
bool dtype_from_safetensors(const std::string& s, DType* out);

struct Tensor {
  void* buf = nullptr;   // MTL::Buffer* (or host pointer in CPU tests, see host)
  void* host = nullptr;  // CPU-visible base pointer of buf (shared storage) or nullptr
  size_t offset = 0;     // bytes from the start of buf
  DType dtype = DType::F32;
  int ndim = 0;
  int64_t shape[4] = {0, 0, 0, 0};

  int64_t numel() const {
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) n *= shape[i];
    return n;
  }
  size_t nbytes() const { return size_t(numel()) * dtype_size(dtype); }
  int64_t dim(int i) const { return shape[i < 0 ? ndim + i : i]; }
  bool valid() const { return buf != nullptr || host != nullptr; }
  template <class T> T* data() const { return reinterpret_cast<T*>(static_cast<char*>(host) + offset); }
  void set_shape(std::initializer_list<int64_t> s) {
    ndim = int(s.size());
    int i = 0;
    for (auto v : s) shape[i++] = v;
  }
  std::string str() const;  // "bf16[3072,2304]"
};

}  // namespace ink
