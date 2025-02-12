#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cuda/std/chrono>
#include <nvbench/nvbench.cuh>

#include "mla_kernel_sm80.cuh"  // IWYU pragma: keep
#include "mla_params.h"
#include "mla_traits_sm80.h"

using namespace llm;

#define DISPATCH_HEAD_DIM_(HEAD_DIM_V, HEAD_DIM_NAME, ...) \
  [&] {                                                    \
    if (HEAD_DIM_V <= 64) {                                \
      constexpr static int HEAD_DIM_NAME = 64;             \
      constexpr static int BLK_N = 64;                     \
      constexpr static int BLK_K = 64;                     \
      return __VA_ARGS__();                                \
    } else if (HEAD_DIM_V <= 128) {                        \
      constexpr static int HEAD_DIM_NAME = 128;            \
      constexpr static int BLK_N = 64;                     \
      constexpr static int BLK_K = 128;                    \
      return __VA_ARGS__();                                \
    } else if (HEAD_DIM_V <= 256) {                        \
      constexpr static int HEAD_DIM_NAME = 256;            \
      constexpr static int BLK_N = 64;                     \
      constexpr static int BLK_K = 128;                    \
      return __VA_ARGS__();                                \
    } else if (HEAD_DIM_V <= 384) {                        \
      constexpr static int HEAD_DIM_NAME = 384;            \
      constexpr static int BLK_N = 64;                     \
      constexpr static int BLK_K = 128;                    \
      return __VA_ARGS__();                                \
    } else if (HEAD_DIM_V <= 512) {                        \
      constexpr static int HEAD_DIM_NAME = 512;            \
      constexpr static int BLK_N = 32;                     \
      constexpr static int BLK_K = 128;                    \
      return __VA_ARGS__();                                \
    } else {                                               \
      assert(false);                                       \
    }                                                      \
  }()

void mla_bench_sm80(nvbench::state& state) {
  // Collect CUPTI metrics
  state.collect_cupti_metrics();

  // Get the parameters
  const auto batch_size = state.get_int64("batch_size");
  const auto q_len = state.get_int64("q_len");
  const auto kv_len = state.get_int64("kv_len");
  const auto n_heads = state.get_int64("n_heads");
  const auto head_dim = state.get_int64("head_dim");
  const auto rope_head_dim = state.get_int64("rope_head_dim");

  const auto options = torch::dtype(torch::kHalf).device(torch::kCUDA);
  const auto q = torch::randn({batch_size, q_len, n_heads, head_dim}, options);
  const auto kv = torch::randn({batch_size, kv_len, head_dim}, options);

  const auto q_rope =
      torch::randn({batch_size, q_len, n_heads, rope_head_dim}, options);
  const auto k_rope =
      torch::randn({batch_size, kv_len, rope_head_dim}, options);

  auto out = torch::empty_like(q);

  // construct attention params
  MLAParams params;
  params.q_ptr = q.const_data_ptr();
  params.q_stride = make_stride(q.stride(0), q.stride(1), q.stride(2));
  params.kv_ptr = kv.const_data_ptr();
  params.kv_stride = make_stride(kv.stride(0), kv.stride(1));

  params.q_rope_ptr = q_rope.const_data_ptr();
  params.q_rope_stride =
      make_stride(q_rope.stride(0), q_rope.stride(1), q_rope.stride(2));
  params.k_rope_ptr = k_rope.const_data_ptr();
  params.k_rope_stride = make_stride(k_rope.stride(0), k_rope.stride(1));

  params.o_ptr = out.mutable_data_ptr();
  params.o_stride = make_stride(out.stride(0), out.stride(1), out.stride(2));

  params.batch_size = batch_size;
  params.max_q_len = q_len;
  params.n_heads = n_heads;
  params.q_len = q_len;
  params.kv_len = kv_len;
  params.head_dim = head_dim;
  params.rope_head_dim = rope_head_dim;
  params.sm_scale = 1.0;
  params.normalize();

  state.exec([&](nvbench::launch& launch) {
    DISPATCH_HEAD_DIM_(head_dim, HEAD_DIM, [&] {
      using Traits = MLATraitsSM80<cute::half_t,
                                   HEAD_DIM,
                                   /*ROPE_HEAD_DIM=*/64,
                                   /*BLK_M=*/64,
                                   BLK_N,
                                   BLK_K>;

      launch_mla_kernel_sm80<Traits>(params, launch.get_stream());
    });
  });
}

NVBENCH_BENCH(mla_bench_sm80)
    .add_int64_axis("batch_size", {1})
    .add_int64_axis("q_len", {1024})
    .add_int64_axis("kv_len", {1024})
    .add_int64_axis("n_heads", {8})
    .add_int64_axis("head_dim", {256})
    .add_int64_axis("rope_head_dim", {64});
