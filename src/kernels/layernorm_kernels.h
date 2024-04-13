#pragma once
#include <torch/torch.h>

namespace llm::kernel {

void rms_norm(torch::Tensor& out,
              torch::Tensor input,
              torch::Tensor weight,
              float epsilon);

void rms_norm_residual(torch::Tensor& out,
                       torch::Tensor& residual,
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

// void invoke_float_layernorm_kernel(float* out,
//                                    const float* input,
//                                    const float* weight,
//                                    const float* bias,
//                                    const float epsilon,
//                                    int m,
//                                    int n);

// void invoke_half2_layernorm_kernel(half2* out,
//                                    const half2* input,
//                                    const half2* weight,
//                                    const half2* bias,
//                                    const float epsilon,
//                                    int m,
//                                    int n);
}  // namespace llm::kernel
