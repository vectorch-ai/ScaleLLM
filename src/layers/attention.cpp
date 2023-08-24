#include "attention.h"

#include <glog/logging.h>
#include <torch/torch.h>

using torch::indexing::Slice;

// ref to flash_attn in third_party/flash_attn
extern std::vector<at::Tensor> mha_varlen_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    c10::optional<at::Tensor>& out_,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    float p_dropout,
    float softmax_scale,
    bool zero_tensors,
    bool is_causal,
    bool return_softmax,
    c10::optional<at::Generator> gen_);

namespace llm::attention {
namespace {
constexpr float negative_infinity = -std::numeric_limits<float>::infinity();

torch::Tensor masked_self_attention(
    torch::Tensor query,  // [num_tokens/seq_len, n_heads, head_dim]
    torch::Tensor key,    // [num_tokens/seq_len, n_heads, head_dim]
    torch::Tensor value,  // [num_tokens/seq_len, n_heads, head_dim]
    torch::Tensor mask,   // [1, seq_len, seq_len]
    float scale) {
  query = query * scale;
  auto scores = torch::einsum("qhd,khd->hqk", {query, key});
  if (mask.defined()) {
    scores += mask;
  }
  // (n_heads, seq_len, seq_len)
  scores = torch::softmax(scores.to(torch::kFloat), /*dim=*/-1).type_as(query);
  return torch::einsum("hqk,khd->qhd", {scores, value});
}
}  // namespace

void varlen_masked_self_attention(
    const torch::Tensor& query,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& key,          // [num_tokens, n_heads, head_dim]
    const torch::Tensor& value,        // [num_tokens, n_heads, head_dim]
    const torch::Tensor& cu_seq_lens,  // [num_seq + 1]
    int32_t max_seq_len,               // maximum sequence length
    torch::Tensor output) {            // [num_tokens, n_heads, head_dim]

  const auto head_dim = query.size(-1);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  c10::optional<at::Tensor> out = output;
  mha_varlen_fwd(query,
                 key,
                 value,
                 out,
                 cu_seq_lens,
                 cu_seq_lens,
                 max_seq_len,
                 max_seq_len,
                 /*p_dropout=*/0.0f,
                 /*softmax_scale=*/scale,
                 /*zero_tensors=*/false,
                 /*is_causal=*/true,
                 /*return_softmax=*/false,
                 /*gen_=*/torch::nullopt);

  // const auto head_dim = query.size(-1);
  // const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  // torch::Tensor cu_seq_lens_cpu = cu_seq_lens.cpu();
  // const size_t num_seqs = cu_seq_lens_cpu.numel() - 1;
  // const int32_t* cu_lens = cu_seq_lens_cpu.data_ptr<int32_t>();
  // std::vector<torch::Tensor> outputs;
  // for (int64_t i = 0; i < num_seqs; ++i) {
  //   // calaculate attention for each sequence
  //   const int32_t start_idx = cu_lens[i];
  //   const int32_t end_idx = cu_lens[i + 1];
  //   const int32_t seq_len = end_idx - start_idx;

  //   // create attention mask based on sequence length
  //   torch::Tensor mask;
  //   if (seq_len > 1) {
  //     mask = torch::full({1, seq_len, seq_len}, negative_infinity);
  //     mask = torch::triu(mask, /*diagonal=*/1).type_as(query);
  //   }
  //   const auto attn = masked_self_attention(
  //       query.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/end_idx),
  //       key.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/end_idx),
  //       value.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/end_idx),
  //       mask,
  //       scale);
  //   output.index_put_({Slice(start_idx, end_idx), Slice(), Slice()}, attn);
  // }
}

void single_token_masked_self_attention(
    const KVCache& kv_cache,     // where to get key and value
    const torch::Tensor& query,  // [num_tokens/num_seq, n_heads, head_dim]
    const torch::Tensor& block_tables,  // [num_tokens, num_blocks]
    const torch::Tensor&
        context_lens,         // [num_tokens] the length of each sequence
    torch::Tensor& output) {  // [num_tokens, n_heads, head_dim]
  const auto num_seq = query.size(0);
  const auto num_heads = query.size(1);
  const auto head_dim = query.size(2);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // process each sequence
  // don't need attention mask for single token
  torch::Tensor mask;
  for (int64_t i = 0; i < num_seq; ++i) {
    // [1, n_heads, head_dim]
    const auto q = query[i].unsqueeze(0);
    const auto block_table = block_tables[i];
    const auto context_len = context_lens[i].item<int>();
    // fetch keys/values from cache
    const auto [k, v] = kv_cache.get_kv_cache(block_table, context_len);
    const auto attn = masked_self_attention(q, k, v, mask, scale);
    output.index_put_({i, Slice(), Slice()}, attn);
  }
}

}  // namespace llm::attention
