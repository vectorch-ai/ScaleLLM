#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cuda/std/chrono>
#include <nvbench/nvbench.cuh>

#include "mha_params.h"
#include "sm80_mha_dispatch.cuh"
#include "sm80_mha_launch.cuh"  // IWYU pragma: keep
#include "static_dispatch.h"

using namespace llm;

void mha_bench_sm80(nvbench::state& state) {
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
  const auto sliding_window = state.get_int64("sliding_window");
  const bool alibi = state.get_int64("alibi") > 0;

  const auto options = torch::dtype(torch::kHalf).device(torch::kCUDA);
  const auto query =
      torch::randn({batch_size, q_len, n_heads, head_dim}, options);
  const auto key =
      torch::randn({batch_size, kv_len, n_kv_heads, head_dim}, options);
  const auto value =
      torch::randn({batch_size, kv_len, n_kv_heads, head_dim}, options);

  auto out = torch::empty_like(query);

  const float sm_scale = 1.0 / sqrt(head_dim);

  torch::optional<torch::Tensor> alibi_slopes;
  if (alibi) {
    alibi_slopes = torch::rand(
        {n_heads}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }

  // construct attention params
  MHAParams params;
  params.q_ptr = query.const_data_ptr();
  params.q_stride =
      make_stride(query.stride(0), query.stride(1), query.stride(2), _1{});
  params.k_ptr = key.const_data_ptr();
  params.k_stride =
      make_stride(key.stride(0), key.stride(1), key.stride(2), _1{});
  params.v_ptr = value.const_data_ptr();
  params.v_stride =
      make_stride(value.stride(0), value.stride(1), value.stride(2), _1{});
  params.o_ptr = out.mutable_data_ptr();
  params.o_stride =
      make_stride(out.stride(0), out.stride(1), out.stride(2), _1{});
  params.alibi_slopes_ptr =
      alibi ? alibi_slopes.value().const_data_ptr<float>() : nullptr;
  params.batch_size = batch_size;
  params.max_q_len = q_len;
  params.n_heads = n_heads;
  params.n_kv_heads = n_kv_heads;
  params.q_len = q_len;
  params.kv_len = kv_len;
  params.head_dim = head_dim;
  params.sm_scale = sm_scale;
  params.logits_soft_cap = logits_soft_cap;
  params.sliding_window = sliding_window;

  state.exec([&](nvbench::launch& launch) {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, [&] {
      sm80_run_mha<cute::half_t, HEAD_DIM>(params, launch.get_stream());
    });
  });
}

NVBENCH_BENCH(mha_bench_sm80)
    .add_int64_axis("batch_size", {1})
    .add_int64_axis("q_len", {1024})
    .add_int64_axis("kv_len", {1024})
    .add_int64_axis("n_heads", {8})
    .add_int64_axis("n_kv_heads", {8})
    .add_int64_axis("head_dim", {64})
    .add_float64_axis("logits_soft_cap", {0.0})
    .add_int64_axis("alibi", {0})
    .add_int64_axis("sliding_window", {-1});
