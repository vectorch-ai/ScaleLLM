#include "stopping_criteria.h"

#include <gflags/gflags_declare.h>

#include <cstdint>
#include <unordered_set>
#include <vector>

DECLARE_int32(num_speculative_tokens);

namespace llm {

namespace {
// Returns whether a given `sequence` ends with `suffix`.
inline bool sequence_end_withs(const Slice<int32_t>& sequence,
                               const Slice<int32_t>& suffix) noexcept {
  return suffix.empty() ||
         (sequence.size() >= suffix.size() &&
          memcmp(sequence.data() + (sequence.size() - suffix.size()),
                 suffix.data(),
                 suffix.size() * sizeof(int32_t)) == 0);
}
}  // namespace

FinishReason StoppingCriteria::check_finished(const Slice<int32_t>& token_ids,
                                              size_t num_prompt_tokens) const {
  CHECK(!token_ids.empty());

  const auto last_token_id = token_ids.back();
  // check against eos token id
  if (!ignore_eos_token && last_token_id == eos_token_id) {
    return FinishReason::STOP;
  }
  // check against stop tokens ids
  if (stop_token_ids.count(last_token_id) > 0) {
    return FinishReason::STOP;
  }

  // check against stop sequences after adding the token
  for (const auto& stop_sequence : stop_sequences) {
    if (stop_sequence.back() == last_token_id &&
        sequence_end_withs(token_ids, stop_sequence)) {
      return FinishReason::STOP;
    }
  }

  // check against max tokens and max context length
  const size_t num_tokens = token_ids.size();
  const bool max_context_length_reached =
      max_context_length > 0 &&
      num_tokens + FLAGS_num_speculative_tokens >= max_context_length;
  CHECK_GE(num_tokens, num_prompt_tokens);
  const size_t num_generated_tokens = num_tokens - num_prompt_tokens;
  const bool max_tokens_reached =
      max_tokens > 0 && num_generated_tokens >= max_tokens;
  if (max_context_length_reached || max_tokens_reached) {
    return FinishReason::LENGTH;
  }
  return FinishReason::NONE;
}

}  // namespace llm
