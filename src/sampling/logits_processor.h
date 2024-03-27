#pragma once
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "kernels/sampling/sampling_kernels.h"
#include "sampling/parameters.h"

namespace llm {
namespace detail {
inline void apply_temperature_penalty(torch::Tensor& logits,
                                      const torch::Tensor& temperatures) {
  // logits: [num_seqs, vocab_size]
  // temperatures: [num_seqs, 1]
  logits.div_(temperatures);
}

inline void apply_repetition_penalty(torch::Tensor& logits,
                                     const torch::Tensor& token_ids,
                                     const torch::Tensor& /*token_ids_lens*/,
                                     const torch::Tensor& penalties) {
  // For now, the padding token (id 0) also gets penalized unexpectedly based on
  // the current implementation.
  // TODO: filter out the padding token ids

  // select the logits for tokens of each sequence
  auto score = logits.gather(/*dim=*/1, /*index=*/token_ids);

  // if score < 0 then repetition penalty has to be multiplied to reduce the
  // previous token probability
  score = torch::where(score < 0, score * penalties, score / penalties);

  // scatter the modified score back to logits
  logits.scatter_(/*dim=*/1, /*index=*/token_ids, /*src=*/score);
}

inline void apply_frequency_presence_penalty(
    torch::Tensor& logits,
    const torch::Tensor& token_ids,
    const torch::Tensor& token_counts,
    const torch::Tensor& /*token_ids_lens*/,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties) {
  // select the logits for tokens of each sequence
  auto score = logits.gather(/*dim=*/1, /*index=*/token_ids);

  // apply frequency and presence penalties
  score.sub_(token_counts * frequency_penalties);
  score.sub_((token_counts > 0) * presence_penalties);

  // scatter the modified score back to logits
  logits.scatter_(/*dim=*/1, /*index=*/token_ids, /*src=*/score);
}
}  // namespace detail

// supported logits processors:
// 1. frequency and presence penalty
// 2. repetition penalty
// 3. temperature

// inspired by transformers LogistProcessor:
// https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L44
// an interface for logits processing that can be used to modify the logits in
// place before sampling
class LogitsProcessor {
 public:
  virtual ~LogitsProcessor() = default;

  // modify the logits in place
  // token_ids: [num_seqs, max_num_tokens]
  // all token ids for each sequence prior to the current generation step
  // used in frequency and presence penalty for now
  // logits: [num_seqs, vocab_size]
  // the logits to be processed
  virtual torch::Tensor forward(const torch::Tensor& logits,
                                const torch::Tensor& token_ids,
                                const torch::Tensor& token_counts,
                                const torch::Tensor& token_ids_lens) const = 0;

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) {
    return this->forward(::std::forward<Args>(args)...);
  }

  // factory method to create a logits processor
  static std::unique_ptr<LogitsProcessor> create(
      const SamplingParameters& params,
      const torch::TensorOptions& options);
};

class LogitsProcessorList : public LogitsProcessor {
 public:
  LogitsProcessorList(std::vector<std::unique_ptr<LogitsProcessor>> processors)
      : processors_(std::move(processors)) {}

  torch::Tensor forward(const torch::Tensor& logits,
                        const torch::Tensor& token_ids,
                        const torch::Tensor& token_counts,
                        const torch::Tensor& token_ids_lens) const override {
    torch::Tensor logits_ = logits;
    for (const auto& processor : processors_) {
      logits_ =
          processor->forward(logits_, token_ids, token_counts, token_ids_lens);
    }
    return logits_;
  }

 private:
  std::vector<std::unique_ptr<LogitsProcessor>> processors_;
};

// https://platform.openai.com/docs/api-reference/parameter-details
// The frequency and presence penalties can be used to reduce the likelihood of
// sampling repetitive sequences of tokens. They work by directly modifying the
// logits (un-normalized log-probabilities) with an additive contribution.
// logits[j] -= c[j] * frequency_penalty + float(c[j] > 0) * presence_penalties
// where c[j] is the number of times the token j has already appeared.
class FrequencyPresencePenaltyLogitsProcessor : public LogitsProcessor {
 public:
  FrequencyPresencePenaltyLogitsProcessor(
      const std::vector<float>& frequency_penalties,
      const std::vector<float>& presence_penalties,
      const torch::TensorOptions& options) {
    frequency_penalties_ =
        torch::tensor(frequency_penalties, options).unsqueeze(1);
    presence_penalties_ =
        torch::tensor(presence_penalties, options).unsqueeze(1);
  }

  torch::Tensor forward(const torch::Tensor& logits,
                        const torch::Tensor& token_ids,
                        const torch::Tensor& token_counts,
                        const torch::Tensor& token_ids_lens) const override {
    torch::Tensor logits_ = logits;
    if (logits_.is_cuda()) {
      kernel::apply_frequency_presence_penalty(logits_,
                                               token_ids,
                                               token_counts,
                                               token_ids_lens,
                                               frequency_penalties_,
                                               presence_penalties_);
    } else {
      detail::apply_frequency_presence_penalty(logits_,
                                               token_ids,
                                               token_counts,
                                               token_ids_lens,
                                               frequency_penalties_,
                                               presence_penalties_);
    }
    return logits_;
  };

 private:
  // the frequency and presence penalties: [num_seqs, 1]
  torch::Tensor frequency_penalties_;
  torch::Tensor presence_penalties_;
};

class RepetitionPenaltyLogitsProcessor : public LogitsProcessor {
 public:
  RepetitionPenaltyLogitsProcessor(const std::vector<float>& penalties,
                                   const torch::TensorOptions& options) {
    penalties_ = torch::tensor(penalties, options).unsqueeze(1);
  }

  // token_ids, [num_seqs, max_num_tokens] LongTensor
  torch::Tensor forward(const torch::Tensor& logits,
                        const torch::Tensor& token_ids,
                        const torch::Tensor& /*token_counts*/,
                        const torch::Tensor& token_ids_lens) const override {
    torch::Tensor logits_ = logits;
    if (logits_.is_cuda()) {
      kernel::apply_repetition_penalty(
          logits_, token_ids, token_ids_lens, penalties_);
    } else {
      detail::apply_repetition_penalty(
          logits_, token_ids, token_ids_lens, penalties_);
    }
    return logits_;
  }

 private:
  // [num_seqs, 1]
  torch::Tensor penalties_;
};

class TemperatureLogitsProcessor : public LogitsProcessor {
 public:
  // Constructor
  // Constructor
  TemperatureLogitsProcessor(const std::vector<float>& temperatures,
                             const torch::TensorOptions& options) {
    // Convert temperature to a tensor and unsqueeze it for broadcasting
    temperatures_ = torch::tensor(temperatures, options).unsqueeze(1);

    // Replace 0. with 1. to avoid division by 0
    temperatures_ =
        torch::where(temperatures_ == 0, torch::tensor(1.0), temperatures_);
  }

  torch::Tensor forward(
      const torch::Tensor& logits,
      const torch::Tensor& /*token_ids*/,
      const torch::Tensor& /*token_counts*/,
      const torch::Tensor& /*token_ids_lens*/) const override {
    torch::Tensor logits_ = logits;
    if (logits_.is_cuda()) {
      kernel::apply_temperature_penalty(logits_, temperatures_);
    } else {
      detail::apply_temperature_penalty(logits_, temperatures_);
    }
    return logits_;
  }

 private:
  torch::Tensor temperatures_;
};

// combine top_k and top_p sampling, apply top_k first then top_p
class TopKTopPLogitsProcessor : public LogitsProcessor {
 public:
  TopKTopPLogitsProcessor(const std::vector<int64_t>& top_k,
                          const std::vector<float>& top_p,
                          const torch::TensorOptions& options) {
    // initialize top_k if any of the values are not 0
    if (std::any_of(
            top_k.begin(), top_k.end(), [](int64_t t) { return t != 0; })) {
      top_k_ = torch::tensor(top_k, options.dtype(torch::kLong)).unsqueeze(1);
      // Replace 0 with max_value to disable top_k
      const auto max_value = std::numeric_limits<int64_t>::max();
      top_k_ = torch::where(top_k_ == 0, torch::tensor(max_value), top_k_);
    }

    // initialize top_p if any of the values are not 1.0
    if (std::any_of(
            top_p.begin(), top_p.end(), [](float t) { return t != 1.0; })) {
      top_p_ = torch::tensor(top_p, options).unsqueeze(1);
    }
  }

  torch::Tensor forward(
      const torch::Tensor& logits,
      const torch::Tensor& /*token_ids*/,
      const torch::Tensor& /*token_counts*/,
      const torch::Tensor& /*token_ids_lens*/) const override {
    // Sort the probabilities in descending order
    auto [logits_sort, logits_idx] =
        logits.sort(/*dim=*/-1, /*descending=*/true);

    const float filter_value = -std::numeric_limits<float>::infinity();
    // ####################  apply top k   ####################
    if (top_k_.defined()) {
      const auto vocab_size = logits.size(-1);
      auto top_k_mask = torch::arange(vocab_size, logits_sort.device())
                            .expand_as(logits_sort);
      top_k_mask = top_k_mask >= top_k_;
      // mask fill the values that are not in the top k
      logits_sort.masked_fill_(top_k_mask, filter_value);
    }

    // ####################  apply top p   ####################
    if (top_p_.defined()) {
      // Calculate the probabilities
      const auto probs_sort = logits_sort.softmax(/*dim=*/-1);
      // Calculate the cumulative sum of sorted probabilities
      const auto probs_sum = probs_sort.cumsum(/*dim=*/-1);
      // Create a mask where (cumulative sum - current value) > p
      const auto mask = (probs_sum - probs_sort) > top_p_;
      // Set values where mask is true to 0.0
      logits_sort.masked_fill_(mask, filter_value);
    }
    return logits_sort.gather(/*dim=*/-1, logits_idx.argsort());
  }

 private:
  torch::Tensor top_k_;
  torch::Tensor top_p_;
};
}  // namespace llm
