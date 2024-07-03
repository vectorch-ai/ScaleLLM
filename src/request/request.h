#pragma once

#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "output.h"
#include "sampling/parameters.h"
#include "sequence.h"
#include "stopping_criteria.h"

namespace llm {

// Function to call when an output is generated.
using OnOutput = std::function<bool(const RequestOutput& output)>;

// A request is a data structure that encapsulates all the necessary
// information required to process a request efficiently. It acts as a
// container, holding essential data, such as input parameters, configuration
// settings, and any additional context-specific details related to the
// request's handling.
struct Request final {
 public:
  // caller needs to gurantee prompt's lifecycle
  Request(std::string prompt,
          std::vector<int32_t> prompt_tokens,
          size_t seq_capacity,
          size_t n,
          size_t best_of,
          bool logprobs);

  void add_sequence();

  bool is_finished() const;

  bool is_streaming() const { return stream; }

  // size_t num_prompt_tokens() const { return prompt_tokens.size(); }

  bool should_expand_sequences() const;

  void expand_sequences();

  void cancel() { is_cancelled_.store(true, std::memory_order_relaxed); }

  bool is_cancelled() const {
    return is_cancelled_.load(std::memory_order_relaxed);
  }

  // Get the elapsed time since the request was created.
  double elapsed_seconds() const {
    return absl::ToDoubleSeconds(absl::Now() - created_time);
  }

  RequestOutput build_output(const Tokenizer& tokenizer);

  // Scheduled time of the request.
  // NOLINTNEXTLINE
  const absl::Time created_time;

  // prompt text string
  // NOLINTNEXTLINE
  const std::string prompt;

  // the token ids from request's prompt.
  // NOLINTNEXTLINE
  const std::vector<int32_t> prompt_tokens;

  // the number of sequences to generate completions for the prompt.
  // NOLINTNEXTLINE
  const size_t n;

  // the number of sequences to generate for each prompt and select the best
  // among.
  const size_t best_of;

  // whether to return log probabilities for output token.
  const bool logprobs;

  // max number of tokens per sequence.
  // NOLINTNEXTLINE
  const size_t seq_capacity;

  // sampling parameters
  SamplingParameter sampling_param;

  // stopping criteria
  StoppingCriteria stopping_criteria;

  // Whether to stream back partial results as they are generated.
  bool stream = false;

  // Whether to echo back the prompt in the output.
  bool echo = false;

  // the priority of the request.
  Priority priority = Priority::NORMAL;

  // list of sequences to generate completions for the prompt
  // use deque instead of vector to avoid no-copy move for Sequence
  std::deque<Sequence> sequences;

  // function to call when an output is generated.
  OnOutput on_output;

 private:
  // is the sequence cancelled
  std::atomic_bool is_cancelled_{false};
};

// Compare two request contexts based on priority then scheduled time.
// if a < b then a should be processed before b.
struct RequestPtrLess {
  bool operator()(const Request* a, const Request* b) const {
    if (a->priority == b->priority) {
      return a->created_time < b->created_time;
    }
    return a->priority < b->priority;
  }
};

// Compare two request contexts based on priority then scheduled time.
// if a > b then a should be processed after b.
struct RequestPtrGreater {
  bool operator()(const Request* a, const Request* b) const {
    if (a->priority == b->priority) {
      return a->created_time > b->created_time;
    }
    return a->priority > b->priority;
  }
};

}  // namespace llm
