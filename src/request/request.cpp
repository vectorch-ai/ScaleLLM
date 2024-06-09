#include "request.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <cstdint>
#include <string>
#include <vector>

#include "request/output.h"
#include "sequence.h"

namespace llm {

Request::Request(std::string prompt,
                 std::vector<int32_t> prompt_tokens,
                 size_t seq_capacity,
                 size_t n,
                 size_t best_of)
    : prompt(std::move(prompt)),
      prompt_tokens(std::move(prompt_tokens)),
      seq_capacity(seq_capacity),
      n(n),
      best_of(best_of),
      created_time(absl::Now()) {
  CHECK_GE(best_of, n);
}

void Request::add_sequence() {
  Sequence::Options options;
  options.echo = this->echo;
  options.sampling_param = this->sampling_param;
  options.stopping_criteria = this->stopping_criteria;

  const size_t index = sequences.size();
  sequences.emplace_back(index,
                         this->prompt,
                         this->prompt_tokens,
                         this->created_time,
                         this->seq_capacity,
                         options);
}

bool Request::is_finished() const {
  // still need to generate more sequences
  if (sequences.size() < best_of) {
    return false;
  }

  return std::all_of(sequences.begin(),
                     sequences.end(),
                     [](const Sequence& seq) { return seq.is_finished(); });
}

bool Request::should_expand_sequences() const {
  if (sequences.size() < best_of) {
    CHECK(!sequences.empty());
    const auto& first_sequence = sequences.front();
    // if all prompt tokens are in kv cache, then expand
    return first_sequence.num_kv_cache_tokens() >=
           first_sequence.num_prompt_tokens();
  }
  return false;
}

void Request::expand_sequences() {
  while (sequences.size() < best_of) {
    add_sequence();
  }
}

RequestOutput Request::build_output(const Tokenizer& tokenizer) {
  // summarize statistics for all sequences
  Usage usage;
  usage.num_prompt_tokens = num_prompt_tokens();
  for (const Sequence& seq : sequences) {
    usage.num_generated_tokens += seq.num_generated_tokens();
  }
  usage.num_total_tokens = usage.num_prompt_tokens + usage.num_generated_tokens;

  RequestOutput output;
  output.usage = usage;

  if (!stream) {
    auto& outputs = output.outputs;
    outputs.reserve(sequences.size());
    // TODO: add logic to select n sequences from best_of sequences
    for (auto& seq : sequences) {
      // generate the final output
      auto seq_output = seq.build_output(tokenizer);
      if (seq_output.has_value()) {
        outputs.push_back(std::move(seq_output.value()));
      }
    }
  }
  output.status = Status(StatusCode::OK);
  output.finished = is_finished();
  return output;
}

}  // namespace llm
