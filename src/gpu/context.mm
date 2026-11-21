// GPU context implementation (Objective-C++, ARC). Owner: B.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu/context.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace ink::gpu {

[[noreturn]] static void die(const std::string& msg) {
  fprintf(stderr, "gpu: %s\n", msg.c_str());
  abort();
}

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static struct BarrierReport {
  long* b = nullptr;
  long* d = nullptr;
  ~BarrierReport() {
    if (b && getenv("INK_GPU_TIME")) fprintf(stderr, "hazard barriers: %ld of %ld dispatches\n", *b, *d);
  }
} g_barrier_report;

// Pipeline plus the set of buffer indices its kernel can write (anything not `const`), from reflection.
static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> dev, id<MTLFunction> f, uint32_t* wmask, NSError** err) {
  MTLComputePipelineReflection* refl = nil;
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f
                                                                   options:MTLPipelineOptionBindingInfo
                                                                reflection:&refl
                                                                     error:err];
  uint32_t m = 0;
  for (id<MTLBinding> b in refl.bindings)
    if (b.type == MTLBindingTypeBuffer && b.access != MTLBindingAccessReadOnly && b.index < 32) m |= 1u << b.index;
  *wmask = p ? m : 0xffffffffu;
  return p;
}

struct Context::Impl {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id residency = nil;  // id<MTLResidencySet> (macOS 15+)
  std::vector<id<MTLLibrary>> libs;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
  id<MTLCommandBuffer> cmd = nil;
  id<MTLComputeCommandEncoder> enc = nil;
  id<MTLCommandBuffer> last = nil;
  std::map<std::string, size_t> by_tag;
  std::vector<id<MTLCommandBuffer>> pending;  // committed early, not yet waited
  // keep-warm: a one-thread kernel on its own queue every 0.5 s for warm_secs after the last wait()
  id<MTLCommandQueue> warm_queue = nil;
  id<MTLComputePipelineState> warm_pipe = nil;
  id<MTLBuffer> warm_buf = nil;
  std::atomic<double> last_activity{0};
  double warm_secs = 0;  // INK_WARM_SECS=300 to enable (off until tested on the GPU)
  std::thread warm_thread;
  int since_commit = 0;
  int commit_every = 128;
  bool null_mode = false;  // INK_NULL=1 (timing experiment: results are garbage)
  id<MTLComputePipelineState> null_pipe = nil;
  id<MTLBuffer> null_buf = nil;
  bool concurrent = false;
  bool untracked = false;
  bool smart = true;  // concurrent encoder + barriers only on buffer hazards (INK_SERIAL=1: serial encoder)
  std::unordered_set<void*> readonly;      // weights: never written by kernels
  std::unordered_set<void*> touched;       // buffers bound by dispatches since the last barrier
  // INK_RW=1: reads and writes tracked separately (write indices from pipeline reflection), so dispatches that only
  // read a common buffer do not force a barrier
  bool rw = false;
  std::unordered_map<void*, uint32_t> wmask;  // pipeline -> writable buffer indices
  uint32_t cur_wmask = 0xffffffffu;
  std::vector<std::pair<uint32_t, void*>> cur_ix;  // (index, buffer) bound for the next dispatch
  std::unordered_set<void*> touched_w;
  std::vector<void*> cur;                  // buffers bound for the next dispatch
  long barriers = 0, dispatches = 0;
  std::unordered_map<void*, std::pair<std::string, size_t>> live;  // mtl ptr -> tag, size
  bool residency_dirty = false;
};

Context& Context::get() {
  static Context ctx;
  return ctx;
}

Context::Context() : impl_(new Impl) {
  @autoreleasepool {
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) die("no Metal device");
    impl_->queue = [impl_->device newCommandQueue];
    device_name_ = [[impl_->device name] UTF8String];
    if (getenv("INK_RESIDENCY") == nullptr || std::string(getenv("INK_RESIDENCY")) != "0") {
      if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
        d.label = @"ink-resident";
        NSError* err = nil;
        id<MTLResidencySet> set = [impl_->device newResidencySetWithDescriptor:d error:&err];
        if (set) {
          impl_->residency = set;
          [impl_->queue addResidencySet:set];
        } else {
          fprintf(stderr, "gpu: residency set unavailable: %s\n", err ? [[err description] UTF8String] : "?");
        }
      }
    }
    // All kernel files form one library, in file-name order (00_common.metal first holds shared helpers).
    std::string all;
    for (auto& [name, src] : embedded_kernels()) all += "// ---- " + name + "\n" + src + "\n";
    if (const char* ce = getenv("INK_COMMIT_EVERY")) impl_->commit_every = atoi(ce);
    if (const char* ws = getenv("INK_WARM_SECS")) impl_->warm_secs = atof(ws);
    impl_->concurrent = getenv("INK_CONCURRENT") != nullptr;
    impl_->null_mode = getenv("INK_NULL") != nullptr;
    impl_->rw = getenv("INK_RW") != nullptr;
    impl_->smart = getenv("INK_SERIAL") == nullptr && !impl_->concurrent;
    g_barrier_report.b = &impl_->barriers;
    g_barrier_report.d = &impl_->dispatches;
    impl_->untracked = getenv("INK_UNTRACKED") != nullptr;
    if (!all.empty()) compile("ink", all);
    // Build every pipeline now, in parallel: lazy creation cost ~2.5 s inside the first request (TTFT).
    if (!getenv("INK_LAZY_PIPELINES") && !impl_->libs.empty()) {
      NSArray<NSString*>* names = [impl_->libs.back() functionNames];
      std::vector<std::string> fn;
      for (NSString* n in names) fn.push_back([n UTF8String]);
      std::vector<id<MTLComputePipelineState>> ps(fn.size());
      id<MTLLibrary> lib = impl_->libs.back();
      id<MTLDevice> dev = impl_->device;
      __strong id<MTLComputePipelineState>* out = ps.data();
      std::vector<uint32_t> masks(fn.size(), 0xffffffffu);
      uint32_t* masks_p = masks.data();
      const std::string* names_p = fn.data();
      dispatch_apply(fn.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        @autoreleasepool {
          id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:names_p[i].c_str()]];
          if (f) out[i] = make_pipeline(dev, f, &masks_p[i], nil);
        }
      });
      for (size_t i = 0; i < fn.size(); i++)
        if (ps[i]) {
          impl_->pipelines[fn[i]] = ps[i];
          impl_->wmask[(__bridge void*)ps[i]] = masks[i];
        }
    }
  }
}

uint64_t Context::recommended_working_set() const { return [impl_->device recommendedMaxWorkingSetSize]; }

void Context::compile(const std::string& name, const std::string& source, const std::string& prefix) {
  @autoreleasepool {
    MTLCompileOptions* opt = [MTLCompileOptions new];
    opt.mathMode = MTLMathModeSafe;  // no fast-math: numerics must match the reference
    opt.languageVersion = MTLLanguageVersion3_1;
    NSError* err = nil;
    std::string full = prefix + source;
    id<MTLLibrary> lib = [impl_->device newLibraryWithSource:[NSString stringWithUTF8String:full.c_str()]
                                                     options:opt
                                                       error:&err];
    if (!lib) die("compile " + name + ":\n" + (err ? [[err description] UTF8String] : "?"));
    impl_->libs.push_back(lib);
  }
}

Pipeline Context::pipeline(const std::string& function) {
  auto it = impl_->pipelines.find(function);
  if (it != impl_->pipelines.end()) return (__bridge void*)it->second;
  @autoreleasepool {
    NSString* fname = [NSString stringWithUTF8String:function.c_str()];
    for (auto it2 = impl_->libs.rbegin(); it2 != impl_->libs.rend(); ++it2) {
      id<MTLFunction> fn = [*it2 newFunctionWithName:fname];
      if (!fn) continue;
      NSError* err = nil;
      uint32_t wm = 0xffffffffu;
      id<MTLComputePipelineState> p = make_pipeline(impl_->device, fn, &wm, &err);
      if (p) impl_->wmask[(__bridge void*)p] = wm;
      if (!p) die("pipeline " + function + ": " + (err ? [[err description] UTF8String] : "?"));
      impl_->pipelines[function] = p;
      return (__bridge void*)p;
    }
  }
  die("no kernel function " + function);
}

uint32_t Context::max_threads(Pipeline p) const {
  return (uint32_t)[(__bridge id<MTLComputePipelineState>)p maxTotalThreadsPerThreadgroup];
}

// ---- memory

Buf Context::alloc(size_t bytes, const char* tag, bool resident) {
  if (bytes == 0) bytes = 16;
  id<MTLBuffer> b = [impl_->device newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared |
                                                       (impl_->untracked ? MTLResourceHazardTrackingModeUntracked
                                                                         : MTLResourceHazardTrackingModeTracked)];
  if (!b) die("alloc " + std::to_string(bytes) + " bytes for " + tag);
  b.label = [NSString stringWithUTF8String:tag];
  if (resident && impl_->residency) {
    if (@available(macOS 15.0, *)) {
      [(id<MTLResidencySet>)impl_->residency addAllocation:b];
      impl_->residency_dirty = true;
    }
  }
  Buf out;
  out.mtl = (__bridge_retained void*)b;
  out.host = [b contents];
  out.size = bytes;
  allocated_ += bytes;
  impl_->by_tag[tag] += bytes;
  impl_->live[out.mtl] = {tag, bytes};
  return out;
}

void Context::release(Buf& b) {
  if (!b.mtl) return;
  id<MTLBuffer> mb = (__bridge_transfer id<MTLBuffer>)b.mtl;
  if (impl_->residency) {
    if (@available(macOS 15.0, *)) {
      [(id<MTLResidencySet>)impl_->residency removeAllocation:mb];
      impl_->residency_dirty = true;
    }
  }
  impl_->readonly.erase(b.mtl);
  auto it = impl_->live.find(b.mtl);
  if (it != impl_->live.end()) {
    impl_->by_tag[it->second.first] -= it->second.second;
    allocated_ -= it->second.second;
    impl_->live.erase(it);
  }
  mb = nil;
  b = Buf{};
}

static void pread_all(const std::string& path, size_t offset, size_t bytes, char* dst) {
  // Split into 64 MB pieces over up to 8 threads; files on the internal SSD read at several GB/s.
  constexpr size_t piece = 64ull << 20;
  size_t n = (bytes + piece - 1) / piece;
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  auto worker = [&] {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      failed = true;
      return;
    }
    fcntl(fd, F_NOCACHE, 1);  // don't keep a second copy in the page cache
    for (size_t i; (i = next++) < n;) {
      size_t off = i * piece, len = std::min(piece, bytes - off);
      size_t done = 0;
      while (done < len) {
        ssize_t r = pread(fd, dst + off + done, len - done, (off_t)(offset + off + done));
        if (r <= 0) {
          failed = true;
          break;
        }
        done += (size_t)r;
      }
    }
    close(fd);
  };
  unsigned nt = (unsigned)std::min<size_t>(n, 8);
  std::vector<std::thread> ts;
  for (unsigned t = 1; t < nt; t++) ts.emplace_back(worker);
  worker();
  for (auto& t : ts) t.join();
  if (failed) die("read failed: " + path);
}

Buf Context::load_file(const std::string& path, size_t offset, size_t bytes, const char* tag) {
  Buf b = alloc(bytes, tag, true);
  impl_->readonly.insert(b.mtl);
  pread_all(path, offset, bytes, (char*)b.host);
  return b;
}

std::vector<Buf> Context::load_files(const std::vector<FileRange>& reqs) {
  std::vector<Buf> out;
  out.reserve(reqs.size());
  for (auto& r : reqs) {
    out.push_back(alloc(r.bytes, r.tag.c_str(), true));
    impl_->readonly.insert(out.back().mtl);
  }
  // Small tensors: one thread per file range from a shared queue; big ones are split inside pread_all.
  std::atomic<size_t> next{0};
  auto worker = [&] {
    for (size_t i; (i = next++) < reqs.size();) {
      if (reqs[i].bytes >= (256ull << 20)) continue;  // big ones below
      pread_all(reqs[i].path, reqs[i].offset, reqs[i].bytes, (char*)out[i].host);
    }
  };
  std::vector<std::thread> ts;
  for (int t = 0; t < 8; t++) ts.emplace_back(worker);
  for (auto& t : ts) t.join();
  for (size_t i = 0; i < reqs.size(); i++)
    if (reqs[i].bytes >= (256ull << 20)) pread_all(reqs[i].path, reqs[i].offset, reqs[i].bytes, (char*)out[i].host);
  return out;
}

void Context::read_file_into(const std::string& path, size_t offset, size_t bytes, Buf dst, size_t dst_offset) {
  if (dst_offset + bytes > dst.size) die("read_file_into: out of range");
  pread_all(path, offset, bytes, (char*)dst.host + dst_offset);
}

void Context::print_memory() const {
  fprintf(stderr, "gpu memory: %.1f MB in all\n", allocated_ / 1e6);
  for (auto& [tag, n] : impl_->by_tag)
    if (n) fprintf(stderr, "  %-24s %10.1f MB\n", tag.c_str(), n / 1e6);
}

// ---- encoding

void Context::ensure_encoder() {
  if (impl_->enc) return;
  if (!impl_->cmd) {
    if (impl_->residency_dirty && impl_->residency) {
      if (@available(macOS 15.0, *)) [(id<MTLResidencySet>)impl_->residency commit];
      impl_->residency_dirty = false;
    }
    impl_->cmd = [impl_->queue commandBufferWithUnretainedReferences];
  }
  impl_->enc = [impl_->cmd computeCommandEncoderWithDispatchType:((impl_->concurrent || impl_->smart)
                                                                                     ? MTLDispatchTypeConcurrent
                                                                                     : MTLDispatchTypeSerial)];
  impl_->touched_w.clear();
  impl_->touched.clear();  // a new encoder starts after the previous one (tracked resources across encoders)
}

void Context::end_encoder() {
  if (impl_->enc) {
    [impl_->enc endEncoding];
    impl_->enc = nil;
  }
}

void Context::set_pipeline(Pipeline p) {
  ensure_encoder();
  [impl_->enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)p];
  auto wm = impl_->wmask.find(p);
  impl_->cur_wmask = wm == impl_->wmask.end() ? 0xffffffffu : wm->second;
}

void Context::set_buffer(uint32_t index, Buf b, size_t offset) {
  ensure_encoder();
  [impl_->enc setBuffer:(__bridge id<MTLBuffer>)b.mtl offset:offset atIndex:index];
  if (impl_->smart && !impl_->readonly.count(b.mtl)) {
    impl_->cur.push_back(b.mtl);
    impl_->cur_ix.push_back({index, b.mtl});
  }
}

void Context::set_bytes(uint32_t index, const void* data, size_t n) {
  ensure_encoder();
  [impl_->enc setBytes:data length:n atIndex:index];
}

void Context::set_threadgroup_memory(uint32_t index, size_t bytes) {
  ensure_encoder();
  [impl_->enc setThreadgroupMemoryLength:(bytes + 15) & ~size_t(15) atIndex:index];
}

void Context::dispatch(Dim3 g, Dim3 t) {
  ensure_encoder();
  if (impl_->smart) {
    // Conservative: any non-weight buffer bound here that a dispatch since the last barrier also bound is a hazard.
    bool hazard = false;
    if (impl_->rw) {
      // later binds at the same index replace earlier ones; a buffer is written if any of its indices is writable
      std::unordered_map<uint32_t, void*> last;
      for (auto& ib : impl_->cur_ix) last[ib.first] = ib.second;
      std::unordered_map<void*, bool> use;  // buffer -> written
      for (auto& kv : last) use[kv.second] = use[kv.second] || (kv.first < 32 && (impl_->cur_wmask >> kv.first) & 1u);
      for (auto& u : use)
        if (impl_->touched_w.count(u.first) || (u.second && impl_->touched.count(u.first))) {
          hazard = true;
          break;
        }
      if (hazard) {
        [impl_->enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        impl_->touched.clear();
        impl_->touched_w.clear();
        impl_->barriers++;
      }
      for (auto& u : use) {
        impl_->touched.insert(u.first);
        if (u.second) impl_->touched_w.insert(u.first);
      }
    } else {
      for (void* b : impl_->cur)
        if (impl_->touched.count(b)) {
          hazard = true;
          break;
        }
      if (hazard) {
        [impl_->enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        impl_->touched.clear();
        impl_->barriers++;
      }
      for (void* b : impl_->cur) impl_->touched.insert(b);
    }
    impl_->cur.clear();
    impl_->cur_ix.clear();
    impl_->dispatches++;
  }
  if (impl_->null_mode) {
    if (!impl_->null_pipe) {
      impl_->null_pipe = (__bridge id<MTLComputePipelineState>)pipeline("null_tick");
      impl_->null_buf = [impl_->device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    }
    [impl_->enc setComputePipelineState:impl_->null_pipe];
    [impl_->enc setBuffer:impl_->null_buf offset:0 atIndex:30];
    [impl_->enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  } else {
    [impl_->enc dispatchThreadgroups:MTLSizeMake(g.x, g.y, g.z) threadsPerThreadgroup:MTLSizeMake(t.x, t.y, t.z)];
  }
  if (impl_->concurrent) [impl_->enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  // Submit in pieces so the GPU starts while the CPU is still encoding the rest of a forward. One serial queue
  // keeps the order; wait() on the last buffer covers the earlier ones.
  if (++impl_->since_commit >= impl_->commit_every && impl_->commit_every > 0) {
    end_encoder();
    [impl_->cmd commit];
    impl_->pending.push_back(impl_->cmd);
    impl_->cmd = nil;
    impl_->since_commit = 0;
  }
}

void Context::barrier() {
  if (impl_->enc) [impl_->enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void Context::blit_copy(Buf src, size_t src_off, Buf dst, size_t dst_off, size_t bytes) {
  end_encoder();
  if (!impl_->cmd) {
    ensure_encoder();
    end_encoder();
  }
  id<MTLBlitCommandEncoder> bl = [impl_->cmd blitCommandEncoder];
  [bl copyFromBuffer:(__bridge id<MTLBuffer>)src.mtl
           sourceOffset:src_off
               toBuffer:(__bridge id<MTLBuffer>)dst.mtl
      destinationOffset:dst_off
                   size:bytes];
  [bl endEncoding];
}

void Context::commit() {
  end_encoder();
  impl_->since_commit = 0;
  if (!impl_->cmd) return;
  [impl_->cmd commit];
  impl_->last = impl_->cmd;
  impl_->cmd = nil;
}

void Context::wait() {
  if (!impl_->last && impl_->pending.empty()) return;
  if (!impl_->last) {  // nothing new since the early commits
    impl_->last = impl_->pending.back();
    impl_->pending.pop_back();
  }
  [impl_->last waitUntilCompleted];
  double first = impl_->last.GPUStartTime;
  for (auto& b : impl_->pending) {
    [b waitUntilCompleted];
    if (b.status == MTLCommandBufferStatusError)
      die(std::string("command buffer failed: ") + [[b.error description] UTF8String]);
    first = std::min(first, b.GPUStartTime);
  }
  if (impl_->last.status == MTLCommandBufferStatusError)
    die(std::string("command buffer failed: ") + [[impl_->last.error description] UTF8String]);
  last_gpu_ms_ = (impl_->last.GPUEndTime - first) * 1e3;  // wall time on the GPU across the pieces
  impl_->last_activity = now_s();
  impl_->pending.clear();
  impl_->last = nil;
}

void Context::commit_and_wait() {
  commit();
  wait();
}

void Context::keep_warm() {
  impl_->last_activity = now_s();
  if (impl_->warm_secs <= 0 || impl_->warm_queue) return;
  impl_->warm_queue = [impl_->device newCommandQueue];
  impl_->warm_pipe = (__bridge id<MTLComputePipelineState>)pipeline("warm_tick");
  impl_->warm_buf = [impl_->device newBufferWithLength:16 options:MTLResourceStorageModeShared];
  Impl* im = impl_;
  im->warm_thread = std::thread([im] {
    while (true) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      if (now_s() - im->last_activity.load() > im->warm_secs) continue;
      @autoreleasepool {
        id<MTLCommandBuffer> cb = [im->warm_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:im->warm_pipe];
        [enc setBuffer:im->warm_buf offset:0 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cb commit];
      }
    }
  });
  im->warm_thread.detach();
}

}  // namespace ink::gpu
