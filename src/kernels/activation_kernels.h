#pragma once
#include <torch/torch.h>

// adapted from https://github.com/NVIDIA/FasterTransformer
namespace llm::kernel {

// clang-format off
template<typename T> struct GeluNewActivation;
template<typename T> struct GeluFastActivation;
template<typename T> struct SiluActivation;
// clang-format on

torch::Tensor gelu_new(torch::Tensor input);
torch::Tensor gelu_fast(torch::Tensor input);
torch::Tensor silu(torch::Tensor input);

}  // namespace llm::kernel
