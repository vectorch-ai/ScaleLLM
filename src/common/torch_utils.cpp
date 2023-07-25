#include "torch_utils.h"

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/torch.h>

namespace llm {

c10::Dict<at::IValue, at::IValue> torch_load_state_dict(
    const std::string& model_path,
    at::DeviceType device_type) {
  torch::Device device(device_type);
  caffe2::serialize::PyTorchStreamReader stream_reader(model_path);
  const auto data =
      torch::jit::readArchiveAndTensors("data",
                                        /*pickle_prefix=*/"",
                                        /*tensor_prefix=*/"",
                                        /*type_resolver=*/c10::nullopt,
                                        /*obj_loader=*/c10::nullopt,
                                        /*device=*/device,
                                        stream_reader);
  return data.toGenericDict();
}

}  // namespace llm
