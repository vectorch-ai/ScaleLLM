// #include <ATen/core/TensorBody.h>
#include <torch/torch.h>
#include <torch/types.h>

// the input tensors are packed into one-dimensional tensors, and the sequence
// lengths are stored in cu_seqlens_q and cu_seqlens_k.
// for each sequence,
// the starting offset: cu_seqlens_q/v[i]
// the length: cu_seqlens_q/v[i+1] - cu_seqlens_q/v[i].
// the maximum sequence length is max_seqlen_q and max_seqlen_k, which are used
// to decide the kernel dispatch.
// clang-format off
std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,         // [n_tokens, n_heads, head_dim]
               const at::Tensor &k,   // [n_tokens, n_kv_heads, head_dim]
               const at::Tensor &v,   // [n_tokens, n_kv_heads, head_dim]
               c10::optional<at::Tensor> &out_, // [n_tokens, n_heads, head_dim]
               const at::Tensor &cu_seqlens_q,  // [batch + 1]
               const at::Tensor &cu_seqlens_k,  // [batch + 1]
               const c10::optional<at::Tensor> &alibi_slopes_, // [num_heads]
               int max_seqlen_q,      // max sequence length for Q
               int max_seqlen_k,      // max sequence length for K/V
               float softmax_scale,
               bool is_causal,
               int window_size_left,
               int window_size_right);
// clang-format on