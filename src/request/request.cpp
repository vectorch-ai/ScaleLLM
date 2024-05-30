#include "request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <vector>

#include "sequence.h"

namespace llm {

Request::Request(std::string prompt,
                 std::vector<int32_t> prompt_tokens,
                 size_t seq_capacity,
                 size_t num_seqs)
    : prompt(std::move(prompt)),
      prompt_tokens(std::move(prompt_tokens)),
      seq_capacity(seq_capacity),
      num_seqs(num_seqs),
      created_time(absl::Now()) {}

void Request::add_sequence() {
  Sequence::Options options;
  options.echo = this->echo;
  options.sampling_param = this->sampling_param;
  options.stopping_criteria = this->stopping_criteria;

  sequences.emplace_back(
      this->prompt, this->prompt_tokens, this->seq_capacity, options);
}

bool Request::is_finished() const {
  // still need to generate more sequences
  if (sequences.size() < num_seqs) {
    return false;
  }

  return std::all_of(sequences.begin(),
                     sequences.end(),
                     [](const Sequence& seq) { return seq.is_finished(); });
}

bool Request::should_expand_sequences() const {
  if (sequences.size() < num_seqs) {
    CHECK(!sequences.empty());
    const auto& first_sequence = sequences.front();
    // if all prompt tokens are in kv cache, then expand
    return first_sequence.num_kv_cache_tokens() >=
           first_sequence.num_prompt_tokens();
  }
  return false;
}

void Request::expand_sequences() {
  while (sequences.size() < num_seqs) {
    add_sequence();
  }
}

}  // namespace llm
