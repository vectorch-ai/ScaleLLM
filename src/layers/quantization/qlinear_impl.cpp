#include "qlinear_impl.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "../model_parallel.h"
#include "model_loader/state_dict.h"
#include "models/args.h"

namespace llm {
namespace {
int64_t round_up(int64_t num, int64_t multiple) {
  return ((num + multiple - 1) / multiple);
}

}  // namespace

ColumnParallelQLinearImpl::ColumnParallelQLinearImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t bits,
    int64_t group_size,
    int64_t qweight_pack_dim,
    int rank,
    int world_size,
    const torch::ScalarType& dtype,
    const torch::Device& device)
    : rank_(rank), world_size_(world_size) {
  CHECK(group_size > 0) << "group_size must be positive";
  CHECK(qweight_pack_dim == 0 || qweight_pack_dim == 1)
      << "qweight_pack_dim must be 0 or 1";
  CHECK(out_features % world_size == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size;
  const int64_t out_features_per_partition = out_features / world_size;
  const int64_t pack_factor = 32 / bits;

  if (qweight_pack_dim == 0) {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({in_features / pack_factor, out_features_per_partition},
                     torch::dtype(torch::kInt32).device(device)),
        /*requires_grad=*/false);
  } else {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({in_features, out_features_per_partition / pack_factor},
                     torch::dtype(torch::kInt32).device(device)),
        /*requires_grad=*/false);
  }
  qzeros_ = register_parameter(
      "qzeros",
      torch::empty({round_up(in_features, group_size),
                    out_features_per_partition / pack_factor},
                   torch::dtype(torch::kInt32).device(device)),
      /*requires_grad=*/false);

  scales_ = register_parameter(
      "scales",
      torch::empty(
          {round_up(in_features, group_size), out_features_per_partition},
          torch::dtype(dtype).device(device)),
      /*requires_grad=*/false);
}

// load the weight from the checkpoint
void ColumnParallelQLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto qweight =
      state_dict.get_sharded_tensor("qweight",
                                    /*dim=*/1,
                                    /*rank=*/rank_,
                                    /*world_size=*/world_size_);
  if (qweight.defined()) {
    CHECK_EQ(qweight_.sizes(), qweight.sizes())
        << "qweight size mismatch for " << name();
    qweight_.copy_(qweight);
    qweight_is_loaded_ = true;
  }
  const auto qzeros = state_dict.get_sharded_tensor("qzeros",
                                                    /*dim=*/1,
                                                    rank_,
                                                    world_size_);
  if (qzeros.defined()) {
    CHECK_EQ(qzeros_.sizes(), qzeros.sizes())
        << "qzeros size mismatch for " << name();
    qzeros_.copy_(qzeros);
    qzeros_is_loaded_ = true;
  }
  const auto scales = state_dict.get_sharded_tensor("scales",
                                                    /*dim=*/1,
                                                    rank_,
                                                    world_size_);
  if (scales.defined()) {
    CHECK_EQ(scales_.sizes(), scales.sizes())
        << "scales size mismatch for " << name();
    scales_.copy_(scales);
    scales_is_loaded_ = true;
  }
}

bool ColumnParallelQLinearImpl::load_weights(
    std::vector<torch::Tensor>& weight_list,
    torch::Tensor& weight) {
  bool all_loaded = std::all_of(
      weight_list.begin(), weight_list.end(), [](const torch::Tensor& t) {
        return t.defined();
      });
  if (!all_loaded) {
    return false;
  }

  auto merged_weight = torch::cat(weight_list, /*dim=*/1);
  // release the memory for weight_list
  weight_list.clear();
  CHECK_EQ(weight.sizes(), merged_weight.sizes())
      << "weight size mismatch for " << name();
  weight.copy_(merged_weight);
  return true;
}

// special load_state_dict for fused cases
void ColumnParallelQLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string_view>& prefixes) {
  if (qweight_list_.size() < prefixes.size()) {
    qweight_list_.resize(prefixes.size());
    qzeros_list_.resize(prefixes.size());
    scales_list_.resize(prefixes.size());
  }

  for (size_t i = 0; i < prefixes.size(); ++i) {
    std::string tensor_name = std::string(prefixes[i]) + "qweight";
    const auto qweight = state_dict.get_sharded_tensor(tensor_name,
                                                       /*dim=*/1,
                                                       rank_,
                                                       world_size_);
    if (qweight.defined()) {
      CHECK(!qweight_list_[i].defined()) << "qweight already loaded";
      // make a copy in case the checkpoint is deleted
      qweight_list_[i] = qweight.clone();
    }
    tensor_name = std::string(prefixes[i]) + "qzeros";
    const auto qzeros = state_dict.get_sharded_tensor(tensor_name,
                                                      /*dim=*/1,
                                                      rank_,
                                                      world_size_);
    if (qzeros.defined()) {
      CHECK(!qzeros_list_[i].defined()) << "qzeros already loaded";
      // make a copy in case the checkpoint is deleted
      qzeros_list_[i] = qzeros.clone();
    }
    tensor_name = std::string(prefixes[i]) + "scales";
    const auto scales = state_dict.get_sharded_tensor(tensor_name,
                                                      /*dim=*/1,
                                                      rank_,
                                                      world_size_);
    if (scales.defined()) {
      CHECK(!scales_list_[i].defined()) << "scales already loaded";
      // make a copy in case the checkpoint is deleted
      scales_list_[i] = scales.clone();
    }
  }

  // check if all weights are ready to be loaded
  if (load_weights(qweight_list_, qweight_)) {
    qweight_is_loaded_ = true;
  }
  if (load_weights(qzeros_list_, qzeros_)) {
    qzeros_is_loaded_ = true;
  }
  if (load_weights(scales_list_, scales_)) {
    scales_is_loaded_ = true;
  }
}

void ColumnParallelQLinearImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(qweight_is_loaded_)
      << "qweight is not loaded for " << prefix + ".qweight";
  CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << prefix + ".qzeros";
  CHECK(scales_is_loaded_) << "scales is not loaded for " << prefix + ".scales";
}

RowParallelQLinearImpl::RowParallelQLinearImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t bits,
    int64_t group_size,
    int64_t qweight_pack_dim,
    int rank,
    int world_size,
    const torch::ScalarType& dtype,
    const torch::Device& device)
    : rank_(rank), world_size_(world_size) {
  CHECK(group_size > 0) << "group_size must be positive";
  CHECK(qweight_pack_dim == 0 || qweight_pack_dim == 1)
      << "qweight_pack_dim must be 0 or 1";
  CHECK(in_features % world_size == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size;
  const int64_t in_features_per_partition = in_features / world_size;
  const int64_t pack_factor = 32 / bits;

  if (qweight_pack_dim == 0) {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({in_features_per_partition / pack_factor, out_features},
                     torch::dtype(torch::kInt32).device(device)),
        /*requires_grad=*/false);
  } else {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({in_features_per_partition, out_features / pack_factor},
                     torch::dtype(torch::kInt32).device(device)),
        /*requires_grad=*/false);
  }
  qzeros_ = register_parameter(
      "qzeros",
      torch::empty({round_up(in_features_per_partition, group_size),
                    out_features / pack_factor},
                   torch::dtype(torch::kInt32).device(device)),
      /*requires_grad=*/false);

  scales_ = register_parameter(
      "scales",
      torch::empty(
          {round_up(in_features_per_partition, group_size), out_features},
          torch::dtype(dtype).device(device)),
      /*requires_grad=*/false);
}

// load the weight from the checkpoint
void RowParallelQLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto qweight = state_dict.get_sharded_tensor("qweight",
                                                     /*dim=*/0,
                                                     rank_,
                                                     world_size_);
  if (qweight.defined()) {
    CHECK_EQ(qweight_.sizes(), qweight.sizes())
        << "qweight size mismatch for " << name();
    qweight_.copy_(qweight);
    qweight_is_loaded_ = true;
  }
  const auto qzeros = state_dict.get_sharded_tensor("qzeros",
                                                    /*dim=*/0,
                                                    rank_,
                                                    world_size_);
  if (qzeros.defined()) {
    CHECK_EQ(qzeros_.sizes(), qzeros.sizes())
        << "qzeros size mismatch for " << name();
    qzeros_.copy_(qzeros);
    qzeros_is_loaded_ = true;
  }
  const auto scales = state_dict.get_sharded_tensor("scales",
                                                    /*dim=*/0,
                                                    rank_,
                                                    world_size_);
  if (scales.defined()) {
    CHECK_EQ(scales_.sizes(), scales.sizes())
        << "scales size mismatch for " << name();
    scales_.copy_(scales);
    scales_is_loaded_ = true;
  }
}

void RowParallelQLinearImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(qweight_is_loaded_)
      << "qweight is not loaded for " << prefix + ".qweight";
  CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << prefix + ".qzeros";
  CHECK(scales_is_loaded_) << "scales is not loaded for " << prefix + ".scales";
}

}  // namespace llm
