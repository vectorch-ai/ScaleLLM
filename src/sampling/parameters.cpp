#include "parameters.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "common/tensor_helper.h"

namespace llm {

void SamplingParameters::init(
    const std::vector<const SamplingParameter*>& sampling_params,
    const std::vector<int32_t>& selected_token_idxes,
    const std::vector<int32_t>& sample_idxes,
    const std::vector<std::vector<int64_t>>& unique_token_ids_vec,
    const std::vector<std::vector<int32_t>>& unique_token_counts_vec,
    const std::vector<int32_t>& unique_token_lens_vec) {
  CHECK_EQ(sampling_params.size(), selected_token_idxes.size());
  CHECK_GE(sampling_params.size(), sample_idxes.size());
  CHECK_EQ(sampling_params.size(), unique_token_ids_vec.size());
  CHECK_EQ(sampling_params.size(), unique_token_counts_vec.size());
  CHECK_EQ(sampling_params.size(), unique_token_lens_vec.size());

  std::vector<float> frequency_penalties;
  std::vector<float> presence_penalties;
  std::vector<float> repetition_penalties;
  std::vector<float> temperatures;
  std::vector<float> top_p;
  std::vector<int64_t> top_k;
  for (const auto* p : sampling_params) {
    frequency_penalties.push_back(p->frequency_penalty);
    presence_penalties.push_back(p->presence_penalty);
    repetition_penalties.push_back(p->repetition_penalty);
    temperatures.push_back(p->temperature);
    top_p.push_back(p->top_p);
    top_k.push_back(p->top_k);
  }

  bool need_token_stats = false;
  if (std::any_of(frequency_penalties.begin(),
                  frequency_penalties.end(),
                  [](float t) { return t != 0.0; })) {
    this->frequency_penalties =
        torch::tensor(frequency_penalties, torch::kFloat32);
    need_token_stats = true;
  }
  if (std::any_of(presence_penalties.begin(),
                  presence_penalties.end(),
                  [](float t) { return t != 0.0; })) {
    this->presence_penalties =
        torch::tensor(presence_penalties, torch::kFloat32);
    need_token_stats = true;
  }
  if (std::any_of(repetition_penalties.begin(),
                  repetition_penalties.end(),
                  [](float t) { return t != 1.0; })) {
    this->repetition_penalties =
        torch::tensor(repetition_penalties, torch::kFloat32);
    need_token_stats = true;
  }
  if (std::any_of(temperatures.begin(), temperatures.end(), [](float t) {
        return t != 0.0 && t != 1.0;
      })) {
    this->temperatures = torch::tensor(temperatures, torch::kFloat32);
  }
  if (std::any_of(
          top_k.begin(), top_k.end(), [](int64_t t) { return t > 0; })) {
    this->top_k = torch::tensor(top_k, torch::kInt64);
  }
  if (std::any_of(
          top_p.begin(), top_p.end(), [](float t) { return t != 1.0; })) {
    this->top_p = torch::tensor(top_p, torch::kFloat32);
  }

  this->selected_token_idxes = torch::tensor(selected_token_idxes, torch::kInt);
  if (need_token_stats) {
    this->unique_token_ids =
        create_2d_tensor(unique_token_ids_vec, torch::kInt64);
    this->unique_token_counts =
        create_2d_tensor(unique_token_counts_vec, torch::kInt);
    this->unique_token_ids_lens =
        torch::tensor(unique_token_lens_vec, torch::kInt);
  }

  // construct do sample tensor
  std::vector<int32_t> do_sample;
  for (const auto idx : sample_idxes) {
    const auto* p = sampling_params[idx];
    // need to do sample if any of following is true
    const bool sample = p->do_sample || p->temperature != 0.0 ||
                        p->top_p != 1.0 || p->top_k > 0;
    do_sample.push_back(sample ? 1 : 0);
  }
  this->sample_idxes = torch::tensor(sample_idxes, torch::kInt);
  this->do_sample = torch::tensor(do_sample, torch::kBool);
}

}  // namespace llm
