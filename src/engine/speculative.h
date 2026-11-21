// Hook for speculative decoding (DSpark). Interface owner A, implementation B (speculative.cpp).
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

#include "engine/engine.h"

namespace ink {

class Model;

// Called by Engine::generate when options().use_mtp. Same contract as Engine::generate (on_token sees every
// emitted token, the last emitted token is not in the cache afterwards). sample_last samples from the model's
// current last logits row. Return false to fall back to plain decoding (nothing emitted yet).
// Called at the end of the Engine constructor when options().use_mtp, before any prefill. B: loads the drafter
// and installs the Model::on_visit capture hook (chaining any previously installed hook).
void speculative_attach(Engine& e, Model& m);

bool speculative_generate(Engine& e, Model& m, const SamplingParams& p, int max_tokens,
                          const std::vector<int32_t>& stop_ids, const std::function<bool(int32_t)>& on_token,
                          const std::function<int32_t()>& sample_last);

}  // namespace ink
