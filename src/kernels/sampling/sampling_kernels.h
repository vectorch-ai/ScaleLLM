#pragma once
#include <torch/torch.h>

namespace llm::kernel {

void apply_temperature_penalty(torch::Tensor& logits,
                               torch::Tensor temperatures);

}  // namespace llm::kernel
