#include "state_dict.h"

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/torch.h>

#include <memory>

namespace llm {

std::unique_ptr<StateDict> StateDict::load_from_file(
    const std::string& model_path,
    torch::DeviceType device_type) {
  using caffe2::serialize::PyTorchStreamReader;

  torch::Device device(device_type);
  PyTorchStreamReader stream_reader(model_path);
  const torch::IValue data =
      torch::jit::readArchiveAndTensors("data",
                                        /*pickle_prefix=*/"",
                                        /*tensor_prefix=*/"",
                                        /*type_resolver=*/c10::nullopt,
                                        /*obj_loader=*/c10::nullopt,
                                        /*device=*/device,
                                        stream_reader);
  // convert to typed dict
  std::unordered_map<std::string, torch::Tensor> dict;
  for (const auto& kv : data.toGenericDict()) {
    const auto& key = kv.key();
    const auto& value = kv.value();
    dict[key.toStringRef()] = value.toTensor();
  }
  return std::make_unique<StateDict>(std::move(dict));
}

torch::Tensor StateDict::get_tensor(const std::string_view& tensor_name) const {
  const auto it = dict_.find(tensor_name.data());
  if (it == dict_.end()) {
    LOG(ERROR) << "Failed to find tensor " << tensor_name;
    return torch::Tensor{nullptr};
  }
  return it->second;
}

// select all the tensors whose name starts with prefix.
StateDict StateDict::select(const std::string_view& prefix) const {
  std::unordered_map<std::string, torch::Tensor> selected;
  for (const auto& [name, tensor] : dict_) {
    std::size_t found = name.find(prefix);
    if (found == 0) {
      selected[name.substr(prefix.length())] = tensor;
    }
  }
  return StateDict(std::move(selected));
}

}  // namespace llm
