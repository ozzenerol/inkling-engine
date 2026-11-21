// Owner: A. Compiles the embedded Metal library and builds the named pipelines; no model, no GPU work.
//   build/kernel_check [kernel names...]   (default: A's kernels)
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "gpu/context.h"

int main(int argc, char** argv) {
  setenv("INK_LAZY_PIPELINES", "1", 1);
  std::vector<std::string> names = {"chain_front", "chain_mid", "ag_scores", "ag_pv", "ag_init", "ag_final", "ag_rel_t"};
  if (argc > 1) names.assign(argv + 1, argv + argc);
  int bad = 0;
  for (auto& n : names) {
    try {
      ink::gpu::Context::get().pipeline(n);
      std::printf("ok   %s\n", n.c_str());
    } catch (const std::exception& e) {
      std::printf("FAIL %s: %s\n", n.c_str(), e.what());
      bad++;
    }
  }
  return bad ? 1 : 0;
}
