#include "state_dict.h"

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/torch.h>

#include <memory>

namespace llm {

namespace {
// adapt from
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/serialization/pickle.cpp#L98
// but with different parameters for efficiency
torch::IValue pickle_load(const std::string& model_path,
                          const torch::Device& device) {
  using caffe2::serialize::PyTorchStreamReader;
  PyTorchStreamReader stream_reader(model_path);
  // add storage context to enable sharing of storage
  auto storage_context =
      std::make_shared<torch::jit::DeserializationStorageContext>();
  return torch::jit::readArchiveAndTensors(
      "data",
      /*pickle_prefix=*/"",
      /*tensor_prefix=*/"",
      /*type_resolver=*/c10::nullopt,
      /*obj_loader=*/c10::nullopt,
      /*device=*/device,
      /*stream_reader=*/stream_reader,
      /*type_parser=*/torch::jit::Unpickler::defaultTypeParser,
      /*storage_context=*/std::move(storage_context));
}

}  // namespace

std::unique_ptr<StateDict> StateDict::load_from_file(
    const std::string& model_path,
    const torch::Device& device) {
  using caffe2::serialize::PyTorchStreamReader;
  LOG(INFO) << "Loading model weights from " << model_path << " to " << device;

  const torch::IValue data = pickle_load(model_path, device);

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
