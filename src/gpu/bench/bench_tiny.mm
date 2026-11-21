// Latency of small dependent kernels (model-free, raw Metal): a chain of N dispatches that ping-pong two
// 2304-float buffers, per kernel shape. Question: what does one small op between the big weight readers cost,
// and which shape (threads per group, number of groups, barriers) is the cheapest for a 2304-wide op?
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

static NSString* kSrc = @R"(
#include <metal_stdlib>
using namespace metal;
// one thread, no work
kernel void k_null(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], uint g [[thread_position_in_grid]]) {
  if (g == 0) out[0] = x[0];
}
// rms as today: one threadgroup of 256, threadgroup-memory reduction with a barrier
kernel void k_rms256(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                     uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[8];
  float ss = 0;
  for (uint i = tid; i < 2304; i += 256) { float v = x[i]; ss += v * v; }
  ss = simd_sum(ss);
  if (lane == 0) scratch[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0;
  for (uint i = 0; i < 8; i++) t += scratch[i];
  float inv = precise::rsqrt(t / 2304.0f + 1e-6f);
  for (uint i = tid; i < 2304; i += 256) out[i] = x[i] * inv;
}
// rms in one simdgroup (32 threads, 72 elements each), no threadgroup memory, no barrier
kernel void k_rms32(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                    uint tid [[thread_position_in_threadgroup]]) {
  float ss = 0;
  for (uint i = tid; i < 2304; i += 32) { float v = x[i]; ss += v * v; }
  ss = simd_sum(ss);
  float inv = precise::rsqrt(ss / 2304.0f + 1e-6f);
  for (uint i = tid; i < 2304; i += 32) out[i] = x[i] * inv;
}
// rms in one threadgroup of 1024
kernel void k_rms1024(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                      uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  float ss = 0;
  for (uint i = tid; i < 2304; i += 1024) { float v = x[i]; ss += v * v; }
  ss = simd_sum(ss);
  if (lane == 0) scratch[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = 0;
  for (uint i = 0; i < 32; i++) t += scratch[i];
  float inv = precise::rsqrt(t / 2304.0f + 1e-6f);
  for (uint i = tid; i < 2304; i += 1024) out[i] = x[i] * inv;
}
// elementwise, one thread per element (grid 2304, groups of G)
kernel void k_ew(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], uint g [[thread_position_in_grid]]) {
  out[g] = x[g] * 0.999f + 0.001f;
}
// elementwise in one simdgroup: 32 threads, 72 elements each
kernel void k_ew32(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], uint tid [[thread_position_in_threadgroup]]) {
  for (uint i = tid; i < 2304; i += 32) out[i] = x[i] * 0.999f + 0.001f;
}
// two ops in one kernel (rms then elementwise), one simdgroup: is a fused pair cheaper than two dispatches?
kernel void k_rms32x2(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                      uint tid [[thread_position_in_threadgroup]]) {
  float ss = 0;
  for (uint i = tid; i < 2304; i += 32) { float v = x[i]; ss += v * v; }
  ss = simd_sum(ss);
  float inv = precise::rsqrt(ss / 2304.0f + 1e-6f);
  float s2 = 0;
  for (uint i = tid; i < 2304; i += 32) { float v = x[i] * inv * 0.999f + 0.001f; s2 += v * v; }
  s2 = simd_sum(s2);
  float inv2 = precise::rsqrt(s2 / 2304.0f + 1e-6f);
  for (uint i = tid; i < 2304; i += 32) out[i] = (x[i] * inv * 0.999f + 0.001f) * inv2;
}
)";

struct Case { const char* name; const char* fn; int grid, group; };

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 4000;
  bool concurrent = getenv("CONCURRENT") != nullptr;  // concurrent encoder + a buffer barrier per dispatch, as the backend
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  NSError* err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:nil error:&err];
  if (!lib) return printf("compile: %s\n", err.localizedDescription.UTF8String), 1;
  id<MTLCommandQueue> q = [dev newCommandQueue];
  id<MTLBuffer> a = [dev newBufferWithLength:2304 * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> b = [dev newBufferWithLength:2304 * 4 options:MTLResourceStorageModeShared];
  const Case cases[] = {{"null (1 thread)", "k_null", 1, 1},
                        {"rms 1 group of 256 + barrier (today)", "k_rms256", 256, 256},
                        {"rms 1 group of 1024 + barrier", "k_rms1024", 1024, 1024},
                        {"rms 1 simdgroup (32)", "k_rms32", 32, 32},
                        {"elementwise 2304 threads, groups of 256", "k_ew", 2304, 256},
                        {"elementwise 2304 threads, groups of 32", "k_ew", 2304, 32},
                        {"elementwise 1 simdgroup (32)", "k_ew32", 32, 32},
                        {"rms+ew+rms fused, 1 simdgroup (32)", "k_rms32x2", 32, 32}};
  for (auto& c : cases) {
    id<MTLComputePipelineState> ps =
        [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(c.fn)] error:&err];
    if (!ps) return printf("pipeline %s: %s\n", c.fn, err.localizedDescription.UTF8String), 1;
    std::vector<double> t;
    for (int r = 0; r < 7; r++) {
      float* pa = (float*)a.contents;
      for (int i = 0; i < 2304; i++) pa[i] = 1.0f + float(i % 7);
      auto t0 = std::chrono::steady_clock::now();
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = concurrent ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                                                  : [cb computeCommandEncoder];
      [e setComputePipelineState:ps];
      for (int i = 0; i < N; i++) {
        [e setBuffer:(i % 2 ? b : a) offset:0 atIndex:0];
        [e setBuffer:(i % 2 ? a : b) offset:0 atIndex:1];
        if (concurrent) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e dispatchThreadgroups:MTLSizeMake(c.grid / c.group, 1, 1) threadsPerThreadgroup:MTLSizeMake(c.group, 1, 1)];
      }
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    printf("%-44s %7.2f us per dispatch (chain of %d, wall, median of 7)\n", c.name, t[3] / N, N);
  }
}
