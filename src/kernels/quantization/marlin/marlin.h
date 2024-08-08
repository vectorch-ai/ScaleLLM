#pragma once
#include <torch/torch.h>

namespace marlin {

void fp16_int4_gemm(const torch::Tensor& A,
                    const torch::Tensor& B,
                    torch::Tensor& C,
                    const torch::Tensor& s,
                    torch::Tensor& workspace,
                    int thread_k = -1,
                    int thread_n = -1,
                    int sms = -1,
                    int max_par = 8);

}  // namespace marlin