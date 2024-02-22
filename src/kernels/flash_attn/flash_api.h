// #include <ATen/core/TensorBody.h>
#include <torch/torch.h>
#include <torch/types.h>

// clang-format off
std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               const c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               const c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               int max_seqlen_k,
               float p_dropout,
               float softmax_scale,
               bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               bool return_softmax,
               c10::optional<at::Generator> gen_);

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                          // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,               // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,               // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                c10::optional<const at::Tensor> &k_,    // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &v_,    // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &seqlens_k_, // batch_size
                c10::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &cache_batch_idx_,  // indices to index into the KV cache
                c10::optional<at::Tensor> &block_table_,            // batch_size x max_num_blocks_per_seq
                c10::optional<at::Tensor> &alibi_slopes_,           // num_heads or batch_size x num_heads
                c10::optional<at::Tensor> &out_,                    // batch_size x seqlen_q x num_heads x head_size
                float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits);

// TODO: Implement the fwd pass for variable length sequences with k/v cache.
std::vector<at::Tensor>
mha_varlen_fwd_kvcache();
// clang-format on