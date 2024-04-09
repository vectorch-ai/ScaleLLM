#include "state_dict.h"

#include <ATen/core/TensorBody.h>
#include <caffe2/serialize/inline_container.h>
#include <glog/logging.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/torch.h>

#include <memory>

#include "huggingface/safetensors.h"

namespace llm {

namespace {
// adapt from
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/serialization/pickle.cpp#L98
// but with different parameters for efficiency
torch::IValue pickle_load(const std::string& model_path) {
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
      /*device=*/c10::nullopt,
      /*stream_reader=*/stream_reader,
      /*type_parser=*/torch::jit::Unpickler::defaultTypeParser,
      /*storage_context=*/std::move(storage_context));
}

torch::ScalarType get_dtype(const Dtype& dtype) {
  switch (dtype) {
    case Dtype::BOOL:
      return torch::kBool;
    case Dtype::U8:
      return torch::kUInt8;
    case Dtype::I8:
      return torch::kInt8;
    case Dtype::I16:
      return torch::kInt16;
    case Dtype::F16:
      return torch::kFloat16;
    case Dtype::BF16:
      return torch::kBFloat16;
    case Dtype::I32:
      return torch::kInt32;
    case Dtype::F32:
      return torch::kFloat32;
    case Dtype::F64:
      return torch::kFloat64;
    case Dtype::I64:
      return torch::kInt64;
    case Dtype::U16:
    case Dtype::U32:
    case Dtype::U64:
    default:
      LOG(FATAL) << "Unsupported dtype " << static_cast<int>(dtype);
  }
  __builtin_unreachable();
}

std::vector<int64_t> get_sizes(const View* view) {
  std::vector<int64_t> sizes;
  sizes.reserve(view->rank);
  for (size_t i = 0; i < view->rank; i++) {
    sizes.push_back(view->shape[i]);
  }
  return sizes;
}

}  // namespace

std::unique_ptr<StateDict> StateDict::load_pickle_file(
    const std::string& weights_file,
    int shard_id,
    int num_shards) {
  CHECK(shard_id >= 0 && shard_id < num_shards)
      << "Invalid shard id " << shard_id << " for " << num_shards << " shards";

  using caffe2::serialize::PyTorchStreamReader;
  const torch::IValue data = pickle_load(weights_file);

  // convert to typed dict
  std::unordered_map<std::string, torch::Tensor> dict;
  for (const auto& kv : data.toGenericDict()) {
    const auto& key = kv.key();
    const auto& value = kv.value();
    dict[key.toStringRef()] = value.toTensor();
  }
  return std::make_unique<StateDict>(std::move(dict), shard_id, num_shards);
}

std::unique_ptr<StateDict> StateDict::load_safetensors(
    const std::string& weights_file,
    int shard_id,
    int num_shards) {
  CHECK(shard_id >= 0 && shard_id < num_shards)
      << "Invalid shard id " << shard_id << " for " << num_shards << " shards";
  folly::MemoryMapping::Options options;
  options.setPrefault(true).setReadable(true);
  auto mem_map = std::make_unique<folly::MemoryMapping>(weights_file.c_str(),
                                                        0,   // offset
                                                        -1,  // length
                                                        options);
  // lock it to memory caused segfault in docker.
  // TODO: reenable it when we figure out the issue.
  // mem_map->mlock(folly::MemoryMapping::LockMode::MUST_LOCK);
  const folly::ByteRange content = mem_map->range();
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  const uint8_t* data = reinterpret_cast<const uint8_t*>(content.data());
  const size_t size = content.size();

  std::unordered_map<std::string, torch::Tensor> dict;
  // safetensors
  Handle* handle = nullptr;
  CHECK(safetensors_deserialize(&handle, data, size) == Status::Ok)
      << "Failed to open safetensors file " << weights_file;

  const char* const* tensor_names = nullptr;
  size_t num_tensors = 0;
  CHECK(safetensors_names(handle, &tensor_names, &num_tensors) == Status::Ok)
      << "Failed to get tensor names from safetensors file " << weights_file;

  for (size_t i = 0; i < num_tensors; i++) {
    const char* tensor_name = tensor_names[i];
    View* tensor_view = nullptr;
    CHECK(safetensors_get_tensor(handle, &tensor_view, tensor_name) ==
          Status::Ok)
        << "Failed to get tensor " << tensor_name << " from safetensors file "
        << weights_file;

    const auto scalar_type = get_dtype(tensor_view->dtype);
    const void* tensor_data = data + tensor_view->start;
    const std::vector<int64_t> tensor_sizes = get_sizes(tensor_view);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const auto tensor = at::from_blob(const_cast<void*>(tensor_data),
                                      tensor_sizes,
                                      torch::dtype(scalar_type));
    CHECK(safetensors_free_tensor(tensor_view) == Status::Ok)
        << "Failed to free tensor view";
    dict[tensor_name] = tensor;
  }
  CHECK(safetensors_free_names(tensor_names, num_tensors) == Status::Ok)
      << "Failed to free tensor names";
  CHECK(safetensors_destroy(handle) == Status::Ok)
      << "Failed to destroy safetensors handle";

  return std::make_unique<StateDict>(
      std::move(mem_map), std::move(dict), shard_id, num_shards);
}

StateDict::StateDict(std::unordered_map<std::string, torch::Tensor> dict,
                     int shard_id,
                     int num_shards)
    : dict_(std::move(dict)), shard_id_(shard_id), num_shards_(num_shards) {
  CHECK(shard_id_ >= 0 && shard_id_ < num_shards_)
      << "Invalid shard id " << shard_id_ << " for " << num_shards_
      << " shards";
}

StateDict::StateDict(std::unique_ptr<folly::MemoryMapping> mem_map,
                     std::unordered_map<std::string, torch::Tensor> dict,
                     int shard_id,
                     int num_shards)
    : mem_map_(std::move(mem_map)),
      dict_(std::move(dict)),
      shard_id_(shard_id),
      num_shards_(num_shards) {
  CHECK(shard_id >= 0 && shard_id < num_shards)
      << "Invalid shard id " << shard_id << " for " << num_shards << " shards";
}

torch::Tensor StateDict::get_tensor(const std::string_view& tensor_name) const {
  const auto it = dict_.find(tensor_name.data());
  if (it == dict_.end()) {
    return torch::Tensor{nullptr};
  }
  // apply transform function if exists
  return transform_func_ ? transform_func_(it->second) : it->second;
}

torch::Tensor StateDict::get_sharded_tensor(const std::string_view& tensor_name,
                                            int64_t dim,
                                            int rank,
                                            int world_size) const {
  CHECK(dim == 0 || dim == 1) << "Only support 1D or 2D sharding";
  CHECK(rank >= 0 && rank < world_size)
      << "Invalid rank " << rank << " for " << world_size << " shards";
  CHECK(world_size >= num_shards_) << "Invalid world size " << world_size
                                   << " for " << num_shards_ << " data shards";
  CHECK(world_size % num_shards_ == 0)
      << "Invalid world size " << world_size << " for " << num_shards_
      << " data shards";
  // check if the tensor contains the data for the given rank
  const int64_t num_ranks_per_shard = world_size / num_shards_;
  const int64_t start_rank = shard_id_ * num_ranks_per_shard;
  const int64_t end_rank = start_rank + num_ranks_per_shard;
  if (rank < start_rank || rank >= end_rank) {
    // not in the range, return empty tensor
    return torch::Tensor{nullptr};
  }

  CHECK(rank >= start_rank && rank < end_rank);
  const int64_t local_rank = rank - start_rank;
  auto tensor = get_tensor(tensor_name);
  if (!tensor.defined()) {
    return tensor;
  }
  // chunk tensor along the dim
  const int64_t dim_size = tensor.size(dim);
  if (dim_size <= num_ranks_per_shard) {
    // too small to shard, return the whole tensor instead
    return tensor;
  }

  CHECK(dim_size % num_ranks_per_shard == 0)
      << "can't devide tensor evenly on " << dim << " with dim: " << dim_size
      << " ranks_per_shard: " << num_ranks_per_shard;
  const auto chunks = tensor.chunk(num_ranks_per_shard, dim);
  return chunks[local_rank];
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
  return {std::move(selected), shard_id_, num_shards_};
}
// in place add the scalar value into the selected tensors ending with suffix
StateDict StateDict::add_scalar_with_suffix(const std::string_view& suffix,
                                            int scalar) const {
  std::unordered_map<std::string, torch::Tensor> selected;
  for (const auto& [name, tensor] : dict_) {
    if (suffix.length() <= name.length() &&
        (name.find(suffix) == (name.length() - suffix.length()))) {
      selected[name] = tensor.clone() + scalar;
    } else {
      selected[name] = tensor.clone();
    }
  }
  return {std::move(selected), shard_id_, num_shards_};
}

StateDict StateDict::select_with_transform(
    const std::string_view& prefix,
    TensorTransform transform_func) const {
  auto selected = select(prefix);
  selected.transform_func_ = std::move(transform_func);
  return selected;
}

}  // namespace llm
