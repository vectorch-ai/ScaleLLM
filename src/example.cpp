#include <gflags/gflags.h>

#include "common/torch_utils.h"

DEFINE_string(model_path, "", "Path to the model file.");

int main(int argc, char* argv[]) {
  // initialize gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const c10::Dict<at::IValue, at::IValue> weights =
      llm::torch_load_state_dict(FLAGS_model_path, torch::kCPU);

  torch::NoGradGuard no_grad;
  for (auto const& w : weights) {
    const std::string& name = w.key().toStringRef();
    const at::Tensor param = w.value().toTensor();
    std::cout << name << ": " << param.sizes() << ", " << param.dtype()
              << std::endl;
  }
  return 0;
}
