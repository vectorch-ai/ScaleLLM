#pragma once

#include <torch/torch.h>
namespace llm {

using ActFunc = torch::Tensor (*)(torch::Tensor);

class Activation {
 public:
  static ActFunc get(const std::string& name);
};

}  // namespace llm
