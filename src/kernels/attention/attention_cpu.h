#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cute/tensor.hpp>

#include "common/range.h"
#include "cute/layout.hpp"
#include "cute/stride.hpp"

namespace llm {
// query/out: [q_len, n_head, head_dim]
// key:   [seq_len, n_kv_head, head_dim]
// value: [seq_len, n_kv_head, head_dim]
inline void mha(torch::Tensor query,
                torch::Tensor key,
                torch::Tensor value,
                torch::Tensor& out) {
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(key.dim(), 3);
  CHECK_EQ(key.sizes(), value.sizes()) << "k and v must have the same shape";
  CHECK_EQ(query.size(2), key.size(2)) << "q and k must have the same head_dim";

  const int64_t q_seq_len = query.size(0);
  const int64_t n_heads = query.size(1);
  const int64_t head_dim = query.size(2);
  const int64_t seq_len = key.size(0);
  const int64_t n_kv_heads = key.size(1);
  CHECK_LE(q_seq_len, seq_len) << "q_len must be less than or equal to seq_len";
  CHECK_LE(n_kv_heads, n_heads)
      << "n_kv_heads must be less than or equal to n_heads";
  CHECK(n_heads % n_kv_heads == 0) << "n_heads must be divisible by n_kv_heads";
  // number of heads to share the same k/v head
  const int64_t group_size = n_heads / n_kv_heads;
  const int64_t q_seq_base = seq_len - q_seq_len;
  const float sm_scale = static_cast<float>(1.0 / std::sqrt(head_dim));

  // convert to cute tensors
  using namespace cute;
  // [q_seq_len, n_heads, head_dim]
  auto g_q = make_tensor(query.data_ptr<float>(),
                         make_shape(q_seq_len, n_heads, head_dim),
                         GenRowMajor{});
  auto g_o = make_tensor(out.data_ptr<float>(),
                         make_shape(q_seq_len, n_heads, head_dim),
                         GenRowMajor{});
  // [seq_len, n_kv_heads, head_dim]
  auto g_k = make_tensor(key.data_ptr<float>(),
                         make_shape(seq_len, n_kv_heads, head_dim),
                         GenRowMajor{});
  auto g_v = make_tensor(value.data_ptr<float>(),
                         make_shape(seq_len, n_kv_heads, head_dim),
                         GenRowMajor{});

  // allocate intermediate storage
  std::vector<float> storage(seq_len);

  // process each query against all key/value in sequence
  for (int64_t q_idx : range(q_seq_len)) {
    // process each head independently
    for (int64_t q_head_idx : range(n_heads)) {
      // corresponding key-value head index
      const int64_t kv_head_idx = q_head_idx / group_size;

      // slice query, key, value, and output for current head
      auto q = g_q(q_idx, q_head_idx, _);                // [head_dim]
      auto o = g_o(q_idx, q_head_idx, _);                // [head_dim]
      auto k = g_k(_, kv_head_idx, _);                   // [seq_len, head_dim]
      auto v = g_v(_, kv_head_idx, _);                   // [seq_len, head_dim]
      auto attn = make_tensor(storage.data(), seq_len);  // [seq_len]
      cute::fill(attn, 0.0f);

      // dot product q and k: s = q * k
      float max = -INFINITY;
      for (int64_t kv_idx : range(seq_len)) {
        for (int64_t d : range(head_dim)) {
          attn(kv_idx) += q(d) * k(kv_idx, d) * sm_scale;
        }
        // apply causal mask
        if (kv_idx > q_seq_base + q_idx) {
          attn(kv_idx) = -INFINITY;
        }
        max = std::max(max, attn(kv_idx));
      }

      // safe softmax: attn = exp(attn - max_score) / sum(...)
      float sum = 0;
      for (int64_t kv_idx : range(seq_len)) {
        attn(kv_idx) = std::exp(attn(kv_idx) - max);
        sum += attn(kv_idx);
      }
      for (int64_t kv_idx : range(seq_len)) {
        attn(kv_idx) /= sum;
      }

      cute::fill(o, 0.0f);
      // dot product s and v: o = s * v
      for (int64_t kv_idx : range(seq_len)) {
        for (int64_t d : range(head_dim)) {
          o[d] += attn(kv_idx) * v(kv_idx, d);
        }
      }
    }
  }
}

}  // namespace llm