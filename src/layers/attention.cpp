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
  query = query * scale;
  auto scores = torch::einsum("bqhd,bkhd->bhqk", {query, key});
  if (mask.defined()) {
    scores += mask;
  }
  // (bs, n_local_heads, seqlen, cache_len + seqlen)
  scores = torch::softmax(scores.to(torch::kFloat), /*dim=*/-1).type_as(query);
  return torch::einsum("bhqk,bkhd->bqhd", {scores, value})
      .reshape({bsz, seqlen, -1});
}

}  // namespace llm
