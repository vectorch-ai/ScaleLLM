#pragma once

#include <torch/torch.h>

namespace marlin {

void fp16_int4_gemm(const torch::Tensor& A,  // (m, k)
                    const torch::Tensor& B,  // (k, n) => (k/16, n*16/8)
                    torch::Tensor& C,        // (m, n)
                    const torch::Tensor& s,  // (n_groups, n)
                    torch::Tensor& workspace,
                    int thread_k = -1,
                    int thread_n = -1,
                    int sms = -1,
                    int max_par = 8);

void gptq_gemm(const torch::Tensor& A,  // (m, k)
               const torch::Tensor& B,  // (k, n) => (k/16, n*16/8)
               torch::Tensor& C,        // (m, n)
               const torch::Tensor& scales,
               const torch::Tensor& zeros,
               const torch::Tensor& g_idx,
               const torch::Tensor& perm,
               torch::Tensor& workspace,
               int num_bits,
               bool is_k_full,
               bool has_zp,
               bool use_fp32_reduce);

void gptq_repack(const torch::Tensor& q_weight,  // (k/pack_factor, n)
                 const torch::Tensor& perm,      // (k)
                 torch::Tensor& out,             // (k/16, n*16/pack_factor)
                 int64_t num_bits);

void awq_repack(const torch::Tensor& q_weight,  // (k, n/pack_factor)
                torch::Tensor& out,             // (k/16, n*16/pack_factor)
                int64_t num_bits);

void fp8_gemm(const torch::Tensor& A,       // (m, k)
              const torch::Tensor& B,       // (k/16, n*16/4)
              torch::Tensor& C,             // (m, n)
              const torch::Tensor& scales,  // (1, n)
              torch::Tensor& workspace);

}  // namespace marlin