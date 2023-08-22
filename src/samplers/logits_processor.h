#pragma once
#include <ATen/core/TensorBody.h>
#include <ATen/ops/any.h>
#include <torch/torch.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace llm {

// supported logits processors:
// 1. frequency and presence penalty
// 2. repetition penalty
// 3. temperature
// 4. top-k
// 5. top-p

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
  virtual torch::Tensor forward(const torch::Tensor& token_ids,
                                const torch::Tensor& logits) const = 0;

  // operator() allows us to use the module as a function.
  template <typename... Args>
  torch::Tensor operator()(Args&&... args) {
    return this->forward(::std::forward<Args>(args)...);
  }
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
      const torch::Dtype& dtype = torch::kFloat32,
      const torch::Device& device = torch::kCPU) {
    torch::TensorOptions options;
    options.dtype(dtype).device(device);
    frequency_penalties_ =
        torch::tensor(frequency_penalties, options).unsqueeze(1);
    presence_penalties_ =
        torch::tensor(presence_penalties, options).unsqueeze(1);
  }

  torch::Tensor forward(const torch::Tensor& token_ids,
                        const torch::Tensor& logits) const override {
    const auto batch_size = logits.size(0);
    const auto vocab_size = logits.size(1);

    // calculate bin count for each sequence
    std::vector<torch::Tensor> bin_counts;
    for (int64_t i = 0; i < batch_size; ++i) {
      bin_counts.push_back(torch::bincount(
          token_ids[i], /*weights=*/{}, /*minlength=*/vocab_size));
    }
    auto bin_counts_tensor = torch::stack(bin_counts, /*dim=*/0);

    // apply frequency and presence penalties
    logits.sub_(bin_counts_tensor * frequency_penalties_);
    logits.sub_((bin_counts_tensor > 0) * presence_penalties_);

    return logits;
  };

 private:
  // the frequency and presence penalties: [num_seqs, 1]
  torch::Tensor frequency_penalties_;
  torch::Tensor presence_penalties_;
};

// Adapted from TGI:
// https://github.com/huggingface/text-generation-inference/blob/main/server/text_generation_server/utils/logits_process.py#L84
class RepetitionPenaltyLogitsProcessor : public LogitsProcessor {
 public:
  RepetitionPenaltyLogitsProcessor(const std::vector<float>& penalties,
                                   const torch::Dtype& dtype = torch::kFloat32,
                                   const torch::Device& device = torch::kCPU) {
    penalties_ =
        torch::tensor(penalties, torch::TensorOptions(dtype).device(device))
            .unsqueeze(1);
  }

  torch::Tensor forward(const torch::Tensor& token_ids,
                        const torch::Tensor& logits) const override {
    // select the logits for tokens of each sequence
    auto score = logits.gather(/*dim=*/1, /*index=*/token_ids);

    // if score < 0 then repetition penalty has to be multiplied to reduce the
    // previous token probability
    score = torch::where(score < 0, score * penalties_, score / penalties_);

    // scatter the modified score back to logits
    logits.scatter_(/*dim=*/1, /*index=*/token_ids, /*src=*/score);
    return logits;
  }

 private:
  // [num_seqs, 1]
  torch::Tensor penalties_;
};

class TemperatureLogitsProcessor : public LogitsProcessor {
 public:
  // Constructor
  TemperatureLogitsProcessor(const std::vector<float>& temperatures,
                             const torch::Dtype& dtype = torch::kFloat32,
                             const torch::Device& device = torch::kCPU) {
    // Convert temperature to a tensor and unsqueeze it for broadcasting
    temperatures_ =
        torch::tensor(temperatures,
                      torch::TensorOptions().dtype(dtype).device(device))
            .unsqueeze(1);

    // Replace 0. with 1. to avoid division by 0
    temperatures_ =
        torch::where(temperatures_ == 0., torch::tensor(1.0), temperatures_);
  }

  torch::Tensor forward(const torch::Tensor& /*token_ids*/,
                        const torch::Tensor& logits) const override {
    logits.div_(temperatures_);
    return logits;
  }

 private:
  torch::Tensor temperatures_;
};

class TopPLogitsProcessor : public LogitsProcessor {
 public:
  TopPLogitsProcessor(
      const std::vector<float>& top_p,
      const torch::Dtype& dtype = torch::kFloat32,
      const torch::Device& device = torch::kCPU,
      float filter_value = -std::numeric_limits<float>::infinity(),
      int min_tokens_to_keep = 1)
      : filter_value(filter_value), min_tokens_to_keep(min_tokens_to_keep) {
    top_p_opposite =
        1.0 -
        torch::tensor(top_p, torch::TensorOptions().dtype(dtype).device(device))
            .unsqueeze(1);
  }

  torch::Tensor forward(const torch::Tensor& /*token_ids*/,
                        const torch::Tensor& logits) const override {
    // sort the logits in descending order
    auto [sorted_logits, sorted_indices] =
        logits.sort(/*dim=*/1, /*descending=*/false);

    // calculate cumulative probabilities
    torch::Tensor cumulative_probs =
        sorted_logits.softmax(/*dim=*/-1).cumsum(/*dim=*/-1);

    // remove tokens with cumulative probability above top_p
    torch::Tensor sorted_indices_to_remove = cumulative_probs <= top_p_opposite;
    sorted_indices_to_remove.index_put_(
        {torch::indexing::Slice(),
         torch::indexing::Slice(-min_tokens_to_keep, torch::indexing::None)},
        false);

    // scatter the modified indices back to logits
    torch::Tensor indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove);
    torch::Tensor warped_scores =
        logits.masked_fill_(indices_to_remove, filter_value);
    return warped_scores;
  }

 private:
  // [num_seqs, 1] FloatTensor
  torch::Tensor top_p_opposite;
  // the value used for filtering, all logits will be set to this value if they
  // are filtered
  float filter_value;
  // the minimum number of tokens to keep
  int min_tokens_to_keep;
};

class TopKLogitsProcessor : public LogitsProcessor {
 public:
  // top_k: input is 1-based, 0 means no filtering or disable filtering
  TopKLogitsProcessor(
      const std::vector<int64_t>& top_k,
      const torch::Device& device = torch::kCPU,
      float filter_value = -std::numeric_limits<float>::infinity(),
      int64_t min_tokens_to_keep = 1)
      : filter_value_(filter_value),
        max_top_k_(*std::max_element(top_k.begin(), top_k.end())),
        min_tokens_to_keep_(min_tokens_to_keep) {
    std::vector<int64_t> adjusted_top_k;
    // need to use int8_t for bool tensor
    // std::vector<bool> is a special case in c++ stl and can't be directly
    // used to initialize a bool tensor
    std::vector<int8_t> disabled;
    adjusted_top_k.reserve(top_k.size());
    disabled.reserve(top_k.size());
    for (auto val : top_k) {
      // adjust top_k to be 0-based
      adjusted_top_k.push_back(std::max(val, min_tokens_to_keep) - 1);
      disabled.push_back(val == 0 ? 1 : 0);
    }

    top_k_ = torch::tensor(
                 adjusted_top_k,
                 torch::TensorOptions().dtype(torch::kInt64).device(device))
                 .unsqueeze(1);

    if (std::any_of(
            disabled.begin(), disabled.end(), [](bool v) { return v == 1; })) {
      top_k_disabled_mask_ =
          torch::tensor(
              disabled,
              torch::TensorOptions().dtype(torch::kBool).device(device))
              .unsqueeze(1);
    }
  }

  torch::Tensor forward(const torch::Tensor& /*token_ids*/,
                        const torch::Tensor& logits) const override {
    torch::Tensor top_k = top_k_;
    auto max_top_k = max_top_k_;

    // if max_top_k > vocab_size, then we need to clamp the top_k values
    const auto vocab_size = logits.size(/*dim=*/-1);
    if (vocab_size < max_top_k_) {
      max_top_k = vocab_size;
      // adjust top_k to be 0-based
      top_k = top_k.clamp_max(/*max=*/vocab_size - 1);
    }

    // get the kth score for each sequence
    auto [topk_scores, _] = logits.topk(/*k=*/max_top_k);
    torch::Tensor kth_scores = topk_scores.gather(/*dim=*/1, /*index=*/top_k);

    if (top_k_disabled_mask_.defined()) {
      // use a very low value for the top-k scores to disable filtering
      kth_scores.masked_fill_(/*mask=*/top_k_disabled_mask_,
                              /*value=*/filter_value_);
    }

    // 'remove' tokens with logits < kth score by setting them to filter_value
    torch::Tensor indices_to_remove = logits < kth_scores;
    logits.masked_fill_(/*mask=*/indices_to_remove, /*value=*/filter_value_);
    return logits;
  }

 private:
  // [num_seqs, 1] IntTensor 0-based
  torch::Tensor top_k_;
  // [num_seqs, 1] BoolTensor to disable top_k filtering for some sequences with
  // top_k = 0
  torch::Tensor top_k_disabled_mask_;
  // the value used for filtering, all logits will be set to this value if they
  // are filtered
  float filter_value_;
  // the maximum value of top_k
  int64_t max_top_k_;
  // the minimum number of tokens to keep
  int64_t min_tokens_to_keep_;
};

}  // namespace llm
