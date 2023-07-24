#include <torch/torch.h>

#include <iostream>

auto main() -> int {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  return 0;
}
