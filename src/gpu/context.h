// GPU context: device, queue, runtime-compiled Metal libraries, resident buffers, command encoding. Owner: B.
// Plain C++ header (no Objective-C types) so every target can include it.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ink::gpu {

// A GPU buffer in shared storage (unified memory: host pointer valid). Non-owning handle; Context owns the memory.
struct Buf {
  void* mtl = nullptr;   // id<MTLBuffer>, bridged
  void* host = nullptr;  // CPU address of offset 0
  size_t size = 0;
  explicit operator bool() const { return mtl != nullptr; }
};

using Pipeline = void*;  // id<MTLComputePipelineState>, bridged

struct Dim3 {
  uint32_t x = 1, y = 1, z = 1;
};

class Context {
 public:
  static Context& get();

  // ---- libraries / pipelines
  // Compile Metal source (prefix is prepended, e.g. "#define X 1\n"). Functions become visible to pipeline().
  void compile(const std::string& name, const std::string& source, const std::string& prefix = "");
  // Pipeline for a kernel function from any compiled library (cached). Aborts if missing.
  Pipeline pipeline(const std::string& function);
  uint32_t max_threads(Pipeline p) const;

  // ---- memory
  // resident=true puts the buffer in the queue's residency set (weights, KV, persistent state).
  Buf alloc(size_t bytes, const char* tag, bool resident = true);
  void release(Buf& b);
  // Read [offset, offset+bytes) of a file into a new resident buffer (parallel pread, no mmap).
  Buf load_file(const std::string& path, size_t offset, size_t bytes, const char* tag);
  // Bulk load: one resident buffer per request, all reads in parallel. Returns buffers in request order.
  struct FileRange {
    std::string path;
    size_t offset = 0, bytes = 0;
    std::string tag;
  };
  std::vector<Buf> load_files(const std::vector<FileRange>& reqs);
  // Read into an existing buffer at dst_offset.
  void read_file_into(const std::string& path, size_t offset, size_t bytes, Buf dst, size_t dst_offset);
  size_t allocated_bytes() const { return allocated_; }
  void print_memory() const;  // totals by tag

  // ---- encoding: one command buffer, one compute encoder, reused across dispatches until commit.
  void set_pipeline(Pipeline p);
  void set_buffer(uint32_t index, Buf b, size_t offset = 0);
  void set_bytes(uint32_t index, const void* data, size_t n);
  template <class T>
  void set_value(uint32_t index, const T& v) { set_bytes(index, &v, sizeof(T)); }
  void set_threadgroup_memory(uint32_t index, size_t bytes);
  void dispatch(Dim3 threadgroups, Dim3 threads_per_group);
  void barrier();  // memory barrier between dispatches (only needed with the concurrent encoder)
  void blit_copy(Buf src, size_t src_off, Buf dst, size_t dst_off, size_t bytes);  // ends compute pass

  void commit();            // submit, don't wait
  void wait();              // wait for last committed
  void commit_and_wait();   // both
  double last_gpu_ms() const { return last_gpu_ms_; }  // GPU time of the last waited command buffer

  const std::string& device_name() const { return device_name_; }
  uint64_t recommended_working_set() const;

 private:
  Context();
  void ensure_encoder();
  void end_encoder();
  struct Impl;
  Impl* impl_;
  size_t allocated_ = 0;
  double last_gpu_ms_ = 0;
  std::string device_name_;
};

// Embedded Metal sources (generated at build time from src/gpu/kernels/*.metal).
const std::vector<std::pair<std::string, std::string>>& embedded_kernels();

}  // namespace ink::gpu
