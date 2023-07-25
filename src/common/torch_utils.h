#pragma once
#include <torch/torch.h>

namespace llm {

c10::Dict<at::IValue, at::IValue> torch_load_state_dict(
    const std::string& model_path,
    at::DeviceType device_type = torch::kCPU);

}  // namespace llm
