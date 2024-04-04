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
                                     const torch::Tensor& unique_token_ids,
                                     const torch::Tensor& /*unique_token_lens*/,
                                     const torch::Tensor& penalties) {
  // For now, the padding token (id 0) also gets penalized unexpectedly based on
  // the current implementation.
  // TODO: filter out the padding token ids

  // select the logits for tokens of each sequence
  auto score = logits.gather(/*dim=*/1, /*index=*/unique_token_ids);

  // if score < 0 then repetition penalty has to be multiplied to reduce the
  // previous token probability
  score = torch::where(score < 0, score * penalties, score / penalties);

  // scatter the modified score back to logits
  logits.scatter_(/*dim=*/1, /*index=*/unique_token_ids, /*src=*/score);
}

inline void apply_frequency_presence_penalty(
    torch::Tensor& logits,
    const torch::Tensor& unique_token_ids,
    const torch::Tensor& unique_token_counts,
    const torch::Tensor& /*unique_token_lens*/,
    const torch::Tensor& frequency_penalties,
    const torch::Tensor& presence_penalties) {
  // select the logits for tokens of each sequence
  auto score = logits.gather(/*dim=*/1, /*index=*/unique_token_ids);

  // apply frequency and presence penalties
  score.sub_(unique_token_counts * frequency_penalties);
  score.sub_((unique_token_counts > 0) * presence_penalties);

  // scatter the modified score back to logits
  logits.scatter_(/*dim=*/1, /*index=*/unique_token_ids, /*src=*/score);
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
  virtual torch::Tensor forward(
      const torch::Tensor& logits,
      const torch::Tensor& unique_token_ids,
      const torch::Tensor& unique_token_counts,
      const torch::Tensor& unique_token_lens) const = 0;

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) {
    return this->forward(::std::forward<Args>(args)...);
  }

  // factory method to create a logits processor
  static std::unique_ptr<LogitsProcessor> create(
      const SamplingParameters& params);
};

class LogitsProcessorList : public LogitsProcessor {
 public:
  LogitsProcessorList(std::vector<std::unique_ptr<LogitsProcessor>> processors)
      : processors_(std::move(processors)) {}

  torch::Tensor forward(const torch::Tensor& logits,
                        const torch::Tensor& unique_token_ids,
                        const torch::Tensor& unique_token_counts,
                        const torch::Tensor& unique_token_lens) const override {
    torch::Tensor logits_ = logits;
    for (const auto& processor : processors_) {
      logits_ = processor->forward(
          logits_, unique_token_ids, unique_token_counts, unique_token_lens);
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
      const torch::Tensor& frequency_penalties,
      const torch::Tensor& presence_penalties) {
    CHECK(frequency_penalties.defined() && presence_penalties.defined());
    frequency_penalties_ = frequency_penalties.unsqueeze(1);
    presence_penalties_ = presence_penalties.unsqueeze(1);
  }

  torch::Tensor forward(const torch::Tensor& logits,
                        const torch::Tensor& unique_token_ids,
                        const torch::Tensor& unique_token_counts,
                        const torch::Tensor& unique_token_lens) const override {
    CHECK_EQ(logits.size(0), frequency_penalties_.size(0));

    torch::Tensor logits_ = logits;
    if (logits_.is_cuda()) {
      kernel::apply_frequency_presence_penalty(logits_,
                                               unique_token_ids,
                                               unique_token_counts,
                                               unique_token_lens,
                                               frequency_penalties_,
                                               presence_penalties_);
    } else {
      detail::apply_frequency_presence_penalty(logits_,
                                               unique_token_ids,
                                               unique_token_counts,
                                               unique_token_lens,
                                               frequency_penalties_,
                                               presence_penalties_);
    }
    return logits_;
  };

 private:
  // the frequency and presence penalties: [num_tokens, 1]
  torch::Tensor frequency_penalties_;
  torch::Tensor presence_penalties_;
};

class RepetitionPenaltyLogitsProcessor : public LogitsProcessor {
 public:
  RepetitionPenaltyLogitsProcessor(const torch::Tensor& penalties) {
    CHECK(penalties.defined());
    penalties_ = penalties.unsqueeze(1);
  }

  // token_ids, [num_seqs, max_num_tokens] LongTensor
  torch::Tensor forward(const torch::Tensor& logits,
                        const torch::Tensor& unique_token_ids,
                        const torch::Tensor& /*unique_token_counts*/,
                        const torch::Tensor& unique_token_lens) const override {
    CHECK_EQ(logits.size(0), penalties_.size(0));
    torch::Tensor logits_ = logits;
    if (logits_.is_cuda()) {
      kernel::apply_repetition_penalty(
          logits_, unique_token_ids, unique_token_lens, penalties_);
    } else {
      detail::apply_repetition_penalty(
          logits_, unique_token_ids, unique_token_lens, penalties_);
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
  TemperatureLogitsProcessor(const torch::Tensor& temperatures) {
    CHECK(temperatures.defined());
    // Convert temperature to a tensor and unsqueeze it for broadcasting
    temperatures_ = temperatures.unsqueeze(1);
  }

  torch::Tensor forward(
      const torch::Tensor& logits,
      const torch::Tensor& /*unique_token_ids*/,
      const torch::Tensor& /*unique_token_counts*/,
      const torch::Tensor& /*unique_token_lens*/) const override {
    CHECK_EQ(logits.size(0), temperatures_.size(0));

    torch::Tensor logits_ = logits;
    if (logits_.is_cuda()) {
      kernel::apply_temperature_penalty(logits_, temperatures_);
    } else {
      detail::apply_temperature_penalty(logits_, temperatures_);
    }
    return logits_;
  }

 private:
  // [n_tokens, 1]
  torch::Tensor temperatures_;
};

// combine top_k and top_p sampling, apply top_k first then top_p
class TopKTopPLogitsProcessor : public LogitsProcessor {
 public:
  TopKTopPLogitsProcessor(const torch::Tensor& top_k,
                          const torch::Tensor& top_p) {
    CHECK(top_k.defined() || top_p.defined());
    if (top_k.defined()) {
      // [n_tokens, 1]
      top_k_ = top_k.unsqueeze(1);
    }

    if (top_p.defined()) {
      // [n_tokens, 1]
      top_p_ = top_p.unsqueeze(1);
    }
  }

  torch::Tensor forward(
      const torch::Tensor& logits,
      const torch::Tensor& /*unique_token_ids*/,
      const torch::Tensor& /*unique_token_counts*/,
      const torch::Tensor& /*unique_token_lens*/) const override {
    // Sort the probabilities in descending order
    auto [logits_sort, logits_idx] =
        logits.sort(/*dim=*/-1, /*descending=*/true);

    const float filter_value = -std::numeric_limits<float>::infinity();
    // ####################  apply top k   ####################
    if (top_k_.defined()) {
      CHECK_EQ(logits.size(0), top_k_.size(0));
      const auto vocab_size = logits.size(-1);
      auto top_k_mask = torch::arange(vocab_size, logits_sort.device())
                            .expand_as(logits_sort);
      top_k_mask = top_k_mask >= top_k_;
      // mask fill the values that are not in the top k
      logits_sort.masked_fill_(top_k_mask, filter_value);
    }

    // ####################  apply top p   ####################
    if (top_p_.defined()) {
      CHECK_EQ(logits.size(0), top_p_.size(0));
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
  // [n_tokens, 1]
  torch::Tensor top_k_;
  // [n_tokens, 1]
  torch::Tensor top_p_;
};
}  // namespace llm
