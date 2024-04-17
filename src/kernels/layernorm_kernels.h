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

template <typename T>
void invoke_layernorm_kernel(T* out,
                             const T* input,
                             const T* weight,
                             const T* bias,
                             const float epsilon,
                             int m,
                             int n);
}  // namespace llm::kernel
