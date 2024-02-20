#pragma once
#include <torch/torch.h>

namespace llm::kernel {

void rms_norm(torch::Tensor& out,    // [...(n_tokens), hidden_size]
              torch::Tensor input,   // [...(n_tokens), hidden_size]
              torch::Tensor weight,  // [hidden_size]
              float epsilon);

void layer_norm(torch::Tensor& out,    // [...(n_tokens), hidden_size]
                torch::Tensor input,   // [...(n_tokens), hidden_size]
                torch::Tensor weight,  // [hidden_size]
                torch::Tensor bias,
                float epsilon);

}  // namespace llm::kernel
