#include "attn_api.h"

#include <ATen/cuda/CUDAContext.h>

#include "cute/layout.hpp"
#include "mha_dispatch_sm80.cuh"
#include "mha_params.h"
#include "static_dispatch.h"

namespace llm {
using namespace cute;

void paged_kv_varlen_mha(
    torch::Tensor& out,                // [n_tokens, n_heads, head_dim]
    const torch::Tensor& query,        // [n_tokens, n_heads, head_dim]
    const torch::Tensor& key_cache,    // [n_slots, n_kv_heads, head_dim]
    const torch::Tensor& value_cache,  // [n_slots, n_kv_heads, head_dim]
    const torch::Tensor& q_cu_lens,    // [batch + 1]
    const torch::Tensor& kv_cu_lens,   // [batch + 1]
    const torch::Tensor& block_table,
    const torch::Tensor& block_cu_lens,                // [batch + 1]
    const std::optional<torch::Tensor>& alibi_slopes,  // [n_heads]
    int block_size,
    int max_q_len,
    int max_kv_len,
    float sm_scale,
    float logits_soft_cap,
    int sliding_window) {
  const int batch_size = q_cu_lens.size(0) - 1;
  const int n_heads = query.size(-2);
  const int n_kv_heads = key_cache.size(-2);
  const int head_dim = query.size(-1);

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
  params.alibi_slopes_ptr = alibi_slopes.has_value()
                                ? alibi_slopes.value().const_data_ptr<float>()
                                : nullptr;
  params.batch_size = batch_size;
  params.block_size = block_size;
  params.max_q_len = max_q_len;
  (void)max_kv_len;  // unused
  params.n_heads = n_heads;
  params.n_kv_heads = n_kv_heads;
  params.head_dim = head_dim;

  params.sm_scale = sm_scale;
  params.logits_soft_cap = logits_soft_cap;
  params.sliding_window = sliding_window;

  params.q_cu_lens = q_cu_lens.const_data_ptr<int32_t>();
  params.kv_cu_lens = kv_cu_lens.const_data_ptr<int32_t>();

  params.block_table = block_table.const_data_ptr<int32_t>();
  params.block_cu_lens = block_cu_lens.const_data_ptr<int32_t>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, [&] {
    DISPATCH_TORCH_DTYPE(query.scalar_type(), DTYPE, [&] {
      run_mha_kernel_sm80<DTYPE, HEAD_DIM>(params, stream);
    });
  });
}

}  // namespace llm
