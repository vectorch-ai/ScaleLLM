#include <absl/random/random.h>
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
  const auto block_size = state.get_int64("block_size");
  const auto q_len = state.get_int64("q_len");
  const auto kv_len = state.get_int64("kv_len");
  const auto n_heads = state.get_int64("n_heads");
  const auto n_kv_heads = state.get_int64("n_kv_heads");
  const auto head_dim = state.get_int64("head_dim");
  const float logits_soft_cap = state.get_float64("logits_soft_cap");
  const auto sliding_window = state.get_int64("sliding_window");
  const bool alibi = state.get_int64("alibi") > 0;

  const int32_t total_blocks = (kv_len * batch_size) / block_size + 2;

  const auto options = torch::dtype(torch::kHalf).device(torch::kCUDA);

  std::vector<int32_t> block_table_vec;
  std::vector<int32_t> block_cu_lens_vec = {0};
  std::vector<int32_t> q_cu_lens_vec = {0};
  std::vector<int32_t> kv_cu_lens_vec = {0};
  int32_t n_kv_tokens = 0;
  int32_t n_q_tokens = 0;
  absl::BitGen gen;
  for (int i = 0; i < batch_size; ++i) {
    n_q_tokens += q_len;
    q_cu_lens_vec.push_back(n_q_tokens);

    n_kv_tokens += kv_len;
    kv_cu_lens_vec.push_back(n_kv_tokens);

    // assign blocks for each sequence randomly
    const int32_t n_blocks = (kv_len + block_size - 1) / block_size;
    std::vector<int32_t> block_bases;
    block_bases.reserve(n_blocks);
    for (int j = 0; j < n_blocks; ++j) {
      // random assign block size
      const int32_t id =
          absl::Uniform<int>(absl::IntervalClosedOpen, gen, 1, total_blocks);
      // put first slot id of each block into block_table
      block_bases.push_back(id * block_size);
    }
    block_table_vec.insert(
        block_table_vec.end(), block_bases.begin(), block_bases.end());
    block_cu_lens_vec.push_back(block_table_vec.size());
  }

  torch::Tensor query = torch::rand({n_q_tokens, n_heads, head_dim}, options);
  const auto n_slots = total_blocks * block_size;
  torch::Tensor key_cache =
      torch::rand({n_slots, n_kv_heads, head_dim}, options);
  torch::Tensor value_cache =
      torch::rand({n_slots, n_kv_heads, head_dim}, options);

  torch::Tensor q_cu_lens = torch::tensor(
      q_cu_lens_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));
  torch::Tensor kv_cu_lens = torch::tensor(
      kv_cu_lens_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));

  torch::Tensor block_table = torch::tensor(
      block_table_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));
  torch::Tensor block_cu_lens = torch::tensor(
      block_cu_lens_vec, torch::dtype(torch::kInt32).device(torch::kCUDA));

  auto out = torch::empty_like(query);

  const float sm_scale = 1.0 / sqrt(head_dim);

  torch::optional<torch::Tensor> alibi_slopes;
  if (alibi) {
    alibi_slopes = torch::rand(
        {n_heads}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }

  // construct attention params
  MHAPagedKVParams params;
  params.q_ptr = query.const_data_ptr();
  params.q_stride = make_stride(query.stride(0), query.stride(1), _1{});
  params.k_ptr = key_cache.const_data_ptr();
  params.k_stride = make_stride(key_cache.stride(0), key_cache.stride(1), _1{});
  params.v_ptr = value_cache.const_data_ptr();
  params.v_stride =
      make_stride(value_cache.stride(0), value_cache.stride(1), _1{});
  params.o_ptr = out.mutable_data_ptr();
  params.o_stride = make_stride(out.stride(0), out.stride(1), _1{});
  params.alibi_slopes_ptr =
      alibi ? alibi_slopes.value().const_data_ptr<float>() : nullptr;
  params.batch_size = batch_size;
  params.max_q_len = q_len;
  params.n_heads = n_heads;
  params.n_kv_heads = n_kv_heads;
  params.head_dim = head_dim;
  params.sm_scale = sm_scale;
  params.logits_soft_cap = logits_soft_cap;
  params.sliding_window = sliding_window;

  params.block_size = block_size;
  params.q_cu_lens = q_cu_lens.const_data_ptr<int32_t>();
  params.kv_cu_lens = kv_cu_lens.const_data_ptr<int32_t>();

  params.block_table = block_table.const_data_ptr<int32_t>();
  params.block_cu_lens = block_cu_lens.const_data_ptr<int32_t>();

  state.exec([&](nvbench::launch& launch) {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, [&] {
      sm80_run_mha<cute::half_t, HEAD_DIM>(params, launch.get_stream());
    });
  });
}

NVBENCH_BENCH(mha_bench_sm80)
    .add_int64_axis("batch_size", {1})
    .add_int64_axis("block_size", {8})
    .add_int64_axis("q_len", {1024})
    .add_int64_axis("kv_len", {1024})
    .add_int64_axis("n_heads", {8})
    .add_int64_axis("n_kv_heads", {8})
    .add_int64_axis("head_dim", {64})
    .add_float64_axis("logits_soft_cap", {0.0})
    .add_int64_axis("alibi", {0})
    .add_int64_axis("sliding_window", {-1});
