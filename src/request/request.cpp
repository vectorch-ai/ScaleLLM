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
                 size_t best_of,
                 bool logprobs)
    : prompt(std::move(prompt)),
      prompt_tokens(std::move(prompt_tokens)),
      seq_capacity(seq_capacity),
      n(n),
      best_of(best_of),
      logprobs(logprobs),
      created_time(absl::Now()) {
  CHECK_GE(best_of, n);
}

void Request::add_sequence() {
  Sequence::Options options;
  options.sampling_param = this->sampling_param;
  options.stopping_criteria = this->stopping_criteria;
  options.echo = this->echo;
  options.logprobs = this->logprobs;

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
  usage.num_prompt_tokens = prompt_tokens.size();
  for (const Sequence& seq : sequences) {
    usage.num_generated_tokens += seq.num_generated_tokens();
  }
  usage.num_total_tokens = usage.num_prompt_tokens + usage.num_generated_tokens;

  RequestOutput output;
  output.usage = usage;
  output.status = Status(StatusCode::OK);
  output.finished = is_finished();

  if (!stream) {
    auto& outputs = output.outputs;
    outputs.reserve(n);
    if (sequences.size() > n) {
      std::vector<std::pair<float, size_t>> sequence_logprobs;
      sequence_logprobs.reserve(sequences.size());
      for (size_t i = 0; i < sequences.size(); ++i) {
        sequence_logprobs.emplace_back(sequences[i].logprob(), i);
      }
      // sort sequences by logprob in descending order
      std::sort(sequence_logprobs.begin(),
                sequence_logprobs.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      // select top n best sequences
      for (size_t i = 0; i < n; ++i) {
        const auto [logprob, index] = sequence_logprobs[i];
        auto seq_output = sequences[index].build_output(tokenizer);
        // override index with the final rank
        seq_output.index = i;
        outputs.push_back(std::move(seq_output));
      }
    } else {
      for (auto& seq : sequences) {
        outputs.push_back(seq.build_output(tokenizer));
      }
    }
  }
  return output;
}

}  // namespace llm
