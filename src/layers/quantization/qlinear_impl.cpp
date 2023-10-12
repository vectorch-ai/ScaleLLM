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
    bool bias,
    const QuantizationArgs& quant_args,
    int64_t qweight_pack_dim,
    bool gather_output,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device)
    : gather_output_(gather_output), parallel_args_(parallel_args) {
  const auto bits = quant_args.bits();
  const auto group_size = quant_args.group_size();
  CHECK(group_size > 0) << "group_size must be positive";
  CHECK(qweight_pack_dim == 0 || qweight_pack_dim == 1)
      << "qweight_pack_dim must be 0 or 1";
  const int64_t world_size = parallel_args.world_size();
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

  scales_ = register_parameter("scales",
                               torch::empty({round_up(in_features, group_size),
                                             out_features_per_partition},
                                            torch::dtype(dtype).device(device)),
                               /*requires_grad=*/false);
  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features_per_partition},
                                            torch::dtype(dtype).device(device)),
                               /*requires_grad=*/false);
  }
}

// load the weight from the checkpoint
void ColumnParallelQLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto qweight =
      state_dict.get_sharded_tensor("qweight",
                                    /*dim=*/1,
                                    parallel_args_.rank(),
                                    parallel_args_.world_size());
  if (qweight.defined()) {
    CHECK(!qweight_is_loaded_) << "qweight already loaded";
    CHECK_EQ(qweight_.sizes(), qweight.sizes())
        << "qweight size mismatch for " << name();
    qweight_.copy_(qweight);
    qweight_is_loaded_ = true;
  }
  const auto qzeros =
      state_dict.get_sharded_tensor("qzeros",
                                    /*dim=*/1,
                                    parallel_args_.rank(),
                                    parallel_args_.world_size());
  if (qzeros.defined()) {
    CHECK(qzeros_.defined()) << "qzeros is not defined for " << name();
    CHECK_EQ(qzeros_.sizes(), qzeros.sizes())
        << "qzeros size mismatch for " << name();
    qzeros_.copy_(qzeros);
    qzeros_is_loaded_ = true;
  }
  const auto scales =
      state_dict.get_sharded_tensor("scales",
                                    /*dim=*/1,
                                    parallel_args_.rank(),
                                    parallel_args_.world_size());
  if (scales.defined()) {
    CHECK(!scales_is_loaded_) << "scales already loaded";
    CHECK_EQ(scales_.sizes(), scales.sizes())
        << "scales size mismatch for " << name();
    scales_.copy_(scales);
    scales_is_loaded_ = true;
  }

  if (bias_.defined()) {
    const auto bias =
        state_dict.get_sharded_tensor("bias",
                                      /*dim=*/0,
                                      parallel_args_.rank(),
                                      parallel_args_.world_size());
    if (bias.defined()) {
      CHECK(!bias_is_loaded_) << "bias already loaded";
      CHECK_EQ(bias_.sizes(), bias.sizes())
          << "bias size mismatch for " << name();
      bias_.copy_(bias);
      bias_is_loaded_ = true;
    }
  }
}

// special load_state_dict for fused cases
void ColumnParallelQLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string_view>& prefixes) {
  const size_t count = prefixes.size();
  std::vector<torch::Tensor> qweight_list(count);
  std::vector<torch::Tensor> qzeros_list(count);
  std::vector<torch::Tensor> scales_list(count);
  std::vector<torch::Tensor> bias_list(count);

  for (size_t i = 0; i < count; ++i) {
    std::string tensor_name = std::string(prefixes[i]) + "qweight";
    const auto qweight =
        state_dict.get_sharded_tensor(tensor_name,
                                      /*dim=*/1,
                                      parallel_args_.rank(),
                                      parallel_args_.world_size());
    if (qweight.defined()) {
      CHECK(!qweight_is_loaded_) << "qweight already loaded";
      CHECK(!qweight_list[i].defined()) << "qweight already loaded";
      qweight_list[i] = qweight;
    }
    tensor_name = std::string(prefixes[i]) + "qzeros";
    const auto qzeros =
        state_dict.get_sharded_tensor(tensor_name,
                                      /*dim=*/1,
                                      parallel_args_.rank(),
                                      parallel_args_.world_size());
    if (qzeros.defined()) {
      CHECK(!qzeros_is_loaded_) << "qzeros already loaded";
      CHECK(!qzeros_list[i].defined()) << "qzeros already loaded";
      qzeros_list[i] = qzeros;
    }
    tensor_name = std::string(prefixes[i]) + "scales";
    const auto scales =
        state_dict.get_sharded_tensor(tensor_name,
                                      /*dim=*/1,
                                      parallel_args_.rank(),
                                      parallel_args_.world_size());
    if (scales.defined()) {
      CHECK(!scales_is_loaded_) << "scales already loaded";
      CHECK(!scales_list[i].defined()) << "scales already loaded";
      scales_list[i] = scales;
    }
    // load bias if defined
    if (bias_.defined()) {
      tensor_name = std::string(prefixes[i]) + "bias";
      const auto bias =
          state_dict.get_sharded_tensor(tensor_name,
                                        /*dim=*/0,
                                        parallel_args_.rank(),
                                        parallel_args_.world_size());
      if (bias.defined()) {
        CHECK(!bias_is_loaded_) << "bias already loaded";
        CHECK(!bias_list[i].defined()) << "bias already loaded";
        bias_list[i] = bias;
      }
    }
  }

  detail::merge_weights(name(),
                        std::move(qweight_list),
                        /*dim=*/1,
                        /*clone=*/true,
                        qweight_list_,
                        qweight_,
                        qweight_is_loaded_);

  detail::merge_weights(name(),
                        std::move(qzeros_list),
                        /*dim=*/1,
                        /*clone=*/true,
                        qzeros_list_,
                        qzeros_,
                        qzeros_is_loaded_);

  detail::merge_weights(name(),
                        std::move(scales_list),
                        /*dim=*/1,
                        /*clone=*/true,
                        scales_list_,
                        scales_,
                        scales_is_loaded_);
  // load bias if defined
  if (bias_.defined()) {
    detail::merge_weights(name(),
                          std::move(bias_list),
                          /*dim=*/0,
                          /*clone=*/true,
                          bias_list_,
                          bias_,
                          bias_is_loaded_);
  }
}

void ColumnParallelQLinearImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(qweight_is_loaded_)
      << "qweight is not loaded for " << prefix + "qweight";
  CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << prefix + "qzeros";
  CHECK(scales_is_loaded_) << "scales is not loaded for " << prefix + "scales";
  CHECK(!bias_.defined() || bias_is_loaded_)
      << "bias is not loaded for " << prefix + "bias";
}

RowParallelQLinearImpl::RowParallelQLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantizationArgs& quant_args,
    int64_t qweight_pack_dim,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    torch::ScalarType dtype,
    const torch::Device& device)
    : input_is_parallelized_(input_is_parallelized),
      parallel_args_(parallel_args) {
  const auto bits = quant_args.bits();
  const auto group_size = quant_args.group_size();
  CHECK(group_size > 0) << "group_size must be positive";
  CHECK(qweight_pack_dim == 0 || qweight_pack_dim == 1)
      << "qweight_pack_dim must be 0 or 1";
  const int64_t world_size = parallel_args.world_size();
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

  if (bias) {
    bias_ = register_parameter(
        "bias",
        torch::empty({out_features}, torch::dtype(dtype).device(device)),
        /*requires_grad=*/false);
  }
}

// load the weight from the checkpoint
void RowParallelQLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto qweight =
      state_dict.get_sharded_tensor("qweight",
                                    /*dim=*/0,
                                    parallel_args_.rank(),
                                    parallel_args_.world_size());
  if (qweight.defined()) {
    CHECK_EQ(qweight_.sizes(), qweight.sizes())
        << "qweight size mismatch for " << name();
    qweight_.copy_(qweight);
    qweight_is_loaded_ = true;
  }
  const auto qzeros =
      state_dict.get_sharded_tensor("qzeros",
                                    /*dim=*/0,
                                    parallel_args_.rank(),
                                    parallel_args_.world_size());
  if (qzeros.defined()) {
    CHECK_EQ(qzeros_.sizes(), qzeros.sizes())
        << "qzeros size mismatch for " << name();
    qzeros_.copy_(qzeros);
    qzeros_is_loaded_ = true;
  }
  const auto scales =
      state_dict.get_sharded_tensor("scales",
                                    /*dim=*/0,
                                    parallel_args_.rank(),
                                    parallel_args_.world_size());
  if (scales.defined()) {
    CHECK_EQ(scales_.sizes(), scales.sizes())
        << "scales size mismatch for " << name();
    scales_.copy_(scales);
    scales_is_loaded_ = true;
  }
  if (bias_.defined()) {
    const auto bias = state_dict.get_tensor("bias");
    if (bias.defined()) {
      CHECK_EQ(bias_.sizes(), bias.sizes())
          << "bias size mismatch for " << name();
      bias_.copy_(bias);
      bias_is_loaded_ = true;
    }
  }
}

void RowParallelQLinearImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(qweight_is_loaded_)
      << "qweight is not loaded for " << prefix + "qweight";
  CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << prefix + "qzeros";
  CHECK(scales_is_loaded_) << "scales is not loaded for " << prefix + "scales";
  CHECK(!bias_.defined() || bias_is_loaded_)
      << "bias is not loaded for " << prefix + "bias";
}

}  // namespace llm
