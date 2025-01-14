#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cuda/std/chrono>
#include <nvbench/nvbench.cuh>

#include "attention_kernel_sm80.cuh"
#include "attention_traits_sm80.h"
#include "kernels/attention/attention_params.h"

using namespace llm;

void attention_bench_sm80(nvbench::state& state) {
  // Collect CUPTI metrics
  state.collect_cupti_metrics();

  // Get the parameters
  const auto batch_size = state.get_int64("batch_size");
  const auto q_len = state.get_int64("q_len");
  const auto kv_len = state.get_int64("kv_len");
  const auto n_heads = state.get_int64("n_heads");
  const auto n_kv_heads = state.get_int64("n_kv_heads");
  const auto head_dim = state.get_int64("head_dim");
  const float logits_soft_cap = state.get_float64("logits_soft_cap");

  const auto options = torch::dtype(torch::kHalf).device(torch::kCUDA);
  const auto query =
      torch::randn({batch_size, n_heads, q_len, head_dim}, options);
  const auto key =
      torch::randn({batch_size, n_kv_heads, kv_len, head_dim}, options);
  const auto value =
      torch::randn({batch_size, n_kv_heads, kv_len, head_dim}, options);

  auto out = torch::empty_like(query);

  const float sm_scale = 1.0 / sqrt(head_dim);
  const auto h_stride = query.stride(1);
  const auto kv_h_stride = key.stride(1);

  constexpr int32_t kHeadDim = 64;
  constexpr int32_t kBlockM = 64;
  constexpr int32_t kBlockN = 64;

  // construct attention params
  AttentionParams params;
  params.q_ptr = query.const_data_ptr();
  params.q_stride =
      make_stride(query.stride(0), query.stride(1), query.stride(2));
  params.k_ptr = key.const_data_ptr();
  params.k_stride = make_stride(key.stride(0), key.stride(1), key.stride(2));
  params.v_ptr = value.const_data_ptr();
  params.v_stride =
      make_stride(value.stride(0), value.stride(1), value.stride(2));
  params.o_ptr = out.mutable_data_ptr();
  params.o_stride = make_stride(out.stride(0), out.stride(1), out.stride(2));
  params.alibi_slopes_ptr = nullptr;
  params.n_heads = n_heads;
  params.n_kv_heads = n_kv_heads;
  params.q_len = q_len;
  params.kv_len = kv_len;
  params.head_dim = head_dim;
  params.sm_scale = sm_scale;
  params.logits_soft_cap = logits_soft_cap;
  params.sliding_window = -1;

  using AttentionTraits = AttentionTraitsSM80<cute::half_t,
                                              kHeadDim,
                                              kBlockM,
                                              kBlockN,
                                              /*Alibi=*/false>;

  dim3 block = AttentionTraits::kThreadNum;
  dim3 grid((q_len + kBlockM - 1) / kBlockM, batch_size, n_heads);

  const auto smem_size = AttentionTraits::kSmemSize;
  auto attention_kernel = mha_kernel_sm80<AttentionTraits, AttentionParams>;

  state.exec([&](nvbench::launch& launch) {
    cudaFuncSetAttribute(attention_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    attention_kernel<<<grid, block, smem_size, launch.get_stream()>>>(params);
  });
}

NVBENCH_BENCH(attention_bench_sm80)
    .add_int64_axis("batch_size", {1})
    .add_int64_axis("q_len", {1024})
    .add_int64_axis("kv_len", {1024})
    .add_int64_axis("n_heads", {8})
    .add_int64_axis("n_kv_heads", {8})
    .add_int64_axis("head_dim", {64})
    .add_float64_axis("logits_soft_cap", {0.0});
