#include <gflags/gflags.h>

#include "common/state_dict.h"

DEFINE_string(model_path, "", "Path to the model file.");

int main(int argc, char* argv[]) {
  // initialize gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const auto weights =
      llm::StateDict::load_from_file(FLAGS_model_path, torch::kCPU);

  torch::NoGradGuard no_grad;
  for (auto const& [name, tensor] : weights) {
    std::cout << name << ": " << tensor.sizes() << ", " << tensor.dtype()
              << std::endl;
  }
  return 0;
}
