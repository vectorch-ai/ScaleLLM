#include "attention.h"

#include <c10/core/TensorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "models/model_args.h"

namespace llm::attention {
namespace {
constexpr float negative_infinity = -std::numeric_limits<float>::infinity();

torch::Tensor masked_self_attention(
    torch::Tensor query,  // [num_tokens, n_heads, head_dim]
    torch::Tensor key,    // [num_tokens, n_heads, head_dim]
    torch::Tensor value,  // [num_tokens, n_heads, head_dim]
    torch::Tensor mask,
    float scale) {
  query = query * scale;
  auto scores = torch::einsum("qhd,khd->hqk", {query, key});
  if (mask.defined()) {
    scores += mask;
  }
  // (bs, n_local_heads, seqlen, cache_len + seqlen)
  scores = torch::softmax(scores.to(torch::kFloat), /*dim=*/-1).type_as(query);
  return torch::einsum("hqk,khd->qhd", {scores, value});
}
}  // namespace

torch::Tensor varlen_masked_self_attention(
    torch::Tensor query,                     // [num_tokens, n_heads, head_dim]
    torch::Tensor key,                       // [num_tokens, n_heads, head_dim]
    torch::Tensor value,                     // [num_tokens, n_heads, head_dim]
    const std::vector<int64_t>& cu_seq_lens  // cumulative sequence lengths
) {
  const auto head_dim = query.size(-1);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const size_t num_seqs = cu_seq_lens.size() - 1;
  std::vector<torch::Tensor> outputs;
  for (size_t i = 0; i < num_seqs; ++i) {
    // calaculate attention for each sequence
    const int64_t start_idx = cu_seq_lens[i];
    const int64_t end_idx = cu_seq_lens[i + 1];
    const int64_t seq_len = end_idx - start_idx;

    // create attention mask based on sequence length
    torch::Tensor mask;
    if (seq_len > 1) {
      mask = torch::full({seq_len, seq_len}, negative_infinity);
      mask = torch::triu(mask, /*diagonal=*/1).type_as(query);
    }
    auto output = masked_self_attention(
        query.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/end_idx),
        key.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/end_idx),
        value.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/end_idx),
        mask,
        scale);
    outputs.push_back(output);
  }

  return torch::cat(outputs, /*dim=*/0);
}

}  // namespace llm::attention
