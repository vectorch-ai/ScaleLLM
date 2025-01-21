#include <torch/torch.h>
#include <torch/types.h>

namespace llm {
// the input tensors are packed into one-dimensional tensors, and the sequence
// lengths are stored in q_cu_lens and k_cu_lens.
// for each sequence,
// the starting offset: q/kv_cu_lens[i]
// the length: q/kv_cu_lens[i+1] - q/kv_cu_lens[i].
// the maximum sequence length is max_q_len and max_kv_len, which are used
// to decide the kernel dispatch.
void paged_kv_varlen_mha(
    torch::Tensor& out,               // [n_tokens, n_heads, head_dim]
    const torch::Tensor& q,           // [n_tokens, n_heads, head_dim]
    const torch::Tensor& k_cache,     // [n_slots, n_kv_heads, head_dim]
    const torch::Tensor& v_cache,     // [n_slots, n_kv_heads, head_dim]
    const torch::Tensor& q_cu_lens,   // [batch + 1]
    const torch::Tensor& kv_cu_lens,  // [batch + 1]
    const torch::Tensor& block_table,
    const torch::Tensor& block_cu_lens,                // [batch + 1]
    const std::optional<torch::Tensor>& alibi_slopes,  // [n_heads]
    int block_size,
    int max_q_len,
    int max_kv_len,
    float sm_scale,
    float logits_soft_cap,
    int sliding_window);

}  // namespace llm