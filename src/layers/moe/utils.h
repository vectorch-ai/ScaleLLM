#include <torch/torch.h>

namespace llm {

std::tuple<torch::Tensor, torch::Tensor> permute(
    const torch::Tensor& tokens,      // [n_tokens, dim]
    const torch::Tensor& routing_map  // [n_experts, n_tokens]
);

torch::Tensor unpermute(
    const torch::Tensor& permuted_tokens,    // [n_permuted_tokens, dim]
    const torch::Tensor& sorted_indices,     // [n_permuted_tokens]
    const torch::IntArrayRef& restore_shape  // [n_tokens, dim]
);

}  // namespace llm