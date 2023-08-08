#pragma once

#include <torch/torch.h>

namespace llm {

// TODO: add detailed logics
// Question: Where to put sampler?
class CausalLMImpl : public torch::nn::Module {
 public:
  // CausalLMImpl(const ModelArgs& args, int64_t world_size);
  // TODO: what is the ouput type?
  torch::Tensor forward(
      const torch::Tensor& tokens,     // [num_tokens]
      const torch::Tensor& positions,  // [num_tokens]
      const torch::Tensor& slots,      // [num_tokens] key value cache slots
      const InputParameters& parameters) {

    // output dim: [num_tokens, vocab_size]
    return torch::zeros({1, 1});
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {}
};
TORCH_MODULE(CausalLM);

}  // namespace llm
