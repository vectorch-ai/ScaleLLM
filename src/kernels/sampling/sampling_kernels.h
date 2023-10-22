#pragma once
#include <torch/torch.h>

namespace llm::kernel {

void apply_temperature_penalty(torch::Tensor& logits,
                               torch::Tensor temperatures);

void apply_repetition_penalty(torch::Tensor& logits,
                              torch::Tensor token_ids,
                              torch::Tensor seq_lens,
                              torch::Tensor penalities);

// calculate softmax in place
void invoke_softmax(torch::Tensor& logits);

}  // namespace llm::kernel
