#include "attention.h"

#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "models/model_args.h"

namespace llm {

torch::Tensor SelfAttentionImpl::forward(
    torch::Tensor query,  // [batch_size, seq_len, n_heads, head_dim]
    torch::Tensor key,    // [batch_size, seq_len, n_heads, head_dim]
    torch::Tensor value,  // [batch_size, seq_len, n_heads, head_dim]
    torch::Tensor mask,
    float scale) const {
  const auto bsz = query.size(0);
  const auto seqlen = query.size(1);

  // (bs, n_local_heads, seqlen, head_dim)
  query = query.transpose(1, 2);
  key = key.transpose(1, 2);
  value = value.transpose(1, 2);

  // [bs, n_local_heads, seqlen, head_dim] x [bs, n_local_heads, head_dim,
  // cache_len + seqlen]
  // => [bs, n_local_heads, seqlen, cache_len + seqlen]
  auto scores = torch::matmul(query, key.transpose(2, 3)) * scale;
  if (mask.defined()) {
    // (bs, n_local_heads, seqlen, cache_len + seqlen)
    scores += mask;
  }
  // (bs, n_local_heads, seqlen, cache_len + seqlen)
  scores = torch::softmax(scores.to(torch::kFloat), -1).type_as(query);
  // (bs, n_local_heads, seqlen, cache_len + seqlen) x [bs, n_local_heads,
  // cache_len + seqlen, head_dim]
  // => [bs, n_local_heads, seqlen, head_dim]
  auto output = torch::matmul(scores, value);
  // (bs, seqlen, dim = n_local_heads X head_dim)
  return output.transpose(1, 2).contiguous().view({bsz, seqlen, -1});
}

}  // namespace llm
