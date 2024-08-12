#pragma once
#include <torch/torch.h>

namespace marlin {

void fp16_int4_gemm(const torch::Tensor& A,  // (m, k)
                    const torch::Tensor& B,  // (k, n) => (k/16, n * 16 / 8)
                    torch::Tensor& C,        // (m, n)
                    const torch::Tensor& s,
                    torch::Tensor& workspace,
                    int thread_k = -1,
                    int thread_n = -1,
                    int sms = -1,
                    int max_par = 8);

torch::Tensor int4_gptq_gemm(
    const torch::Tensor& A,  // (m, k)
    const torch::Tensor& B,  // (k, n) => (k/16, n * 16 / 8)
    torch::Tensor& C,        // (m, n)
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    const torch::Tensor& g_idx,
    const torch::Tensor& perm,
    torch::Tensor& workspace,
    bool is_k_full,
    bool has_zp,
    bool use_fp32_reduce);

torch::Tensor fp16_int8_gptq_gemm(
    const torch::Tensor& A,  // (m, k)
    const torch::Tensor& B,  // (k, n) => (k/16, n * 16 / 4)
    torch::Tensor& C,        // (m, n)
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    const torch::Tensor& g_idx,
    const torch::Tensor& perm,
    torch::Tensor& workspace,
    bool is_k_full,
    bool has_zp,
    bool use_fp32_reduce);

}  // namespace marlin