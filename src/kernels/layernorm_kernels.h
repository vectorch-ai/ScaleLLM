#pragma once
#include <torch/torch.h>

namespace llm::kernel {

void rms_norm(torch::Tensor& out,
              torch::Tensor input,
              torch::Tensor weight,
              float epsilon);
              
void layer_norm(torch::Tensor& out,
                torch::Tensor input,
                torch::Tensor weight,
                torch::Tensor bias,
                float epsilon);

}  // namespace llm::kernel
