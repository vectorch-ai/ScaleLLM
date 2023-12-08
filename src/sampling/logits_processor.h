#pragma once
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "kernels/sampling/sampling_kernels.h"
#include "request/sampling_parameter.h"

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
  virtual void forward(torch::Tensor& logits,
                       const torch::Tensor& token_ids,
                       const torch::Tensor& token_counts,
                       const torch::Tensor& token_ids_lens) const = 0;

  // operator() allows us to use the module as a function.
  template <typename... Args>
  void operator()(Args&&... args) {
    this->forward(::std::forward<Args>(args)...);
  }

  // factory method to create a logits processor
  static std::unique_ptr<LogitsProcessor> create(
      const SamplingParameters& params,
      torch::ScalarType dtype,
      const torch::Device& device);
};

class LogitsProcessorList : public LogitsProcessor {
 public:
  LogitsProcessorList(std::vector<std::unique_ptr<LogitsProcessor>> processors)
      : processors_(std::move(processors)) {}

  void forward(torch::Tensor& logits,
               const torch::Tensor& token_ids,
               const torch::Tensor& token_counts,
               const torch::Tensor& token_ids_lens) const override {
    for (const auto& processor : processors_) {
      processor->forward(logits, token_ids, token_counts, token_ids_lens);
    }
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
      torch::ScalarType dtype,
      const torch::Device& device) {
    frequency_penalties_ =
        torch::tensor(frequency_penalties, torch::dtype(dtype).device(device))
            .unsqueeze(1);
    presence_penalties_ =
        torch::tensor(presence_penalties, torch::dtype(dtype).device(device))
            .unsqueeze(1);
  }

  void forward(torch::Tensor& logits,
               const torch::Tensor& token_ids,
               const torch::Tensor& token_counts,
               const torch::Tensor& token_ids_lens) const override {
    if (logits.is_cuda()) {
      kernel::apply_frequency_presence_penalty(logits,
                                               token_ids,
                                               token_counts,
                                               token_ids_lens,
                                               frequency_penalties_,
                                               presence_penalties_);
    } else {
      detail::apply_frequency_presence_penalty(logits,
                                               token_ids,
                                               token_counts,
                                               token_ids_lens,
                                               frequency_penalties_,
                                               presence_penalties_);
    }
  };

 private:
  // the frequency and presence penalties: [num_seqs, 1]
  torch::Tensor frequency_penalties_;
  torch::Tensor presence_penalties_;
};

class RepetitionPenaltyLogitsProcessor : public LogitsProcessor {
 public:
  RepetitionPenaltyLogitsProcessor(const std::vector<float>& penalties,
                                   torch::ScalarType dtype,
                                   const torch::Device& device) {
    penalties_ = torch::tensor(penalties, torch::dtype(dtype).device(device))
                     .unsqueeze(1);
  }

  // token_ids, [num_seqs, max_num_tokens] LongTensor
  void forward(torch::Tensor& logits,
               const torch::Tensor& token_ids,
               const torch::Tensor& /*token_counts*/,
               const torch::Tensor& token_ids_lens) const override {
    if (logits.is_cuda()) {
      kernel::apply_repetition_penalty(
          logits, token_ids, token_ids_lens, penalties_);
    } else {
      detail::apply_repetition_penalty(
          logits, token_ids, token_ids_lens, penalties_);
    }
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
                             const torch::Dtype& dtype = torch::kFloat32,
                             const torch::Device& device = torch::kCPU) {
    // Convert temperature to a tensor and unsqueeze it for broadcasting
    temperatures_ =
        torch::tensor(temperatures, torch::dtype(dtype).device(device))
            .unsqueeze(1);

    // Replace 0. with 1. to avoid division by 0
    temperatures_ =
        torch::where(temperatures_ == 0, torch::tensor(1.0), temperatures_);
  }

  void forward(torch::Tensor& logits,
               const torch::Tensor& /*token_ids*/,
               const torch::Tensor& /*token_counts*/,
               const torch::Tensor& /*token_ids_lens*/) const override {
    if (logits.is_cuda()) {
      kernel::apply_temperature_penalty(logits, temperatures_);
    } else {
      detail::apply_temperature_penalty(logits, temperatures_);
    }
  }

 private:
  torch::Tensor temperatures_;
};

}  // namespace llm
