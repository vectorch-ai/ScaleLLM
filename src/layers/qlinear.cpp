#include "qlinear.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "model_loader/state_dict.h"
#include "model_parallel.h"
#include "models/parallel_args.h"

extern void vecquant2matmul_cuda(torch::Tensor vec,
                                 torch::Tensor mat,
                                 torch::Tensor mul,
                                 torch::Tensor scales,
                                 torch::Tensor zeros,
                                 torch::Tensor g_idx);

extern void vecquant3matmul_cuda(torch::Tensor vec,
                                 torch::Tensor mat,
                                 torch::Tensor mul,
                                 torch::Tensor scales,
                                 torch::Tensor zeros,
                                 torch::Tensor g_idx);

extern void vecquant4matmul_cuda(torch::Tensor vec,
                                 torch::Tensor mat,
                                 torch::Tensor mul,
                                 torch::Tensor scales,
                                 torch::Tensor zeros,
                                 torch::Tensor g_idx);

extern void vecquant8matmul_cuda(torch::Tensor vec,
                                 torch::Tensor mat,
                                 torch::Tensor mul,
                                 torch::Tensor scales,
                                 torch::Tensor zeros,
                                 torch::Tensor g_idx);

namespace llm {
namespace {
void vec_quant_matmul_cuda(torch::Tensor vec,
                           torch::Tensor mat,
                           torch::Tensor mul,
                           torch::Tensor scales,
                           torch::Tensor zeros,
                           torch::Tensor g_idx,
                           int64_t bits) {
  switch (bits) {
    case 2:
      return vecquant2matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
    case 3:
      return vecquant3matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
    case 4:
      return vecquant4matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
    case 8:
      return vecquant8matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
    default:
      LOG(FATAL) << "Unsupported bits " << bits;
  }
  __builtin_unreachable();
}

int64_t num_ints(int64_t count, int64_t bits) {
  return (bits == 2) ? 16 : (bits == 3) ? 11 : (bits == 4) ? 8 : 4;
}

int64_t round_up(int64_t num, int64_t multiple) {
  return ((num + multiple - 1) / multiple);
}

}  // namespace

ColumnParallelQuantLinearImpl::ColumnParallelQuantLinearImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t bits,
    int64_t group_size,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::ScalarType& /*dtype*/,
    const torch::Device& device)
    : in_features_(in_features),
      out_features_(out_features),
      bits_(bits),
      group_size_(group_size),
      parallel_args_(parallel_args),
      gather_output_(gather_output) {
  CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 8)
      << "Only 2,3,4,8 bits are supported";
  CHECK(group_size > 0) << "group_size must be positive";
  CHECK(in_features % 32 == 0) << "in_features must be divisible by 32";
  CHECK(in_features % group_size == 0) << "in_features must be divisible by "
                                       << group_size;
  CHECK(out_features % 32 == 0) << "out_features must be divisible by 32";


  const auto world_size = parallel_args_.world_size();
  CHECK(out_features % world_size == 0)
      << "out_features " << out_features << " not divisible by world_size "
      << world_size;
  const int64_t out_features_per_partition = out_features / world_size;

  // TODO: support partioning
  qweight_ = register_parameter(
      "qweight",
      torch::empty({in_features / 32 * bits, out_features_per_partition},
                   torch::dtype(torch::kInt32).device(device)),
      /*requires_grad=*/false);
  qzeros_ = register_parameter(
      "qzeros",
      torch::empty({round_up(in_features, group_size),
                    out_features_per_partition / 32 * bits},
                   torch::dtype(torch::kInt32).device(device)),
      /*requires_grad=*/false);

  scales_ = register_parameter(
      "scales",
      torch::empty(
          {round_up(in_features, group_size), out_features_per_partition},
          torch::dtype(torch::kFloat16).device(device)),
      /*requires_grad=*/false);

  std::vector<int> g_idx_data;
  for (int i = 0; i < in_features; ++i) {
    g_idx_data.push_back(i / group_size);
  }
  g_idx_ = register_buffer(
      "g_idx",
      torch::tensor(g_idx_data, torch::dtype(torch::kInt32).device(device)));
}

torch::Tensor ColumnParallelQuantLinearImpl::forward(torch::Tensor input) {
  auto output =
      torch::empty({input.size(0), out_features_},
                   torch::dtype(torch::kFloat32).device(input.device()));
  vec_quant_matmul_cuda(
      input, qweight_, output, scales_, qzeros_, g_idx_, bits_);
  if (parallel_args_.world_size() > 1 && gather_output_) {
    output = gather_from_model_parallel_region(output, parallel_args_);
  }
  return output.to(input);
}

// load the weight from the checkpoint
void ColumnParallelQuantLinearImpl::load_state_dict(
    const StateDict& state_dict) {
  const auto qweight =
      state_dict.get_sharded_tensor("qweight",
                                    /*dim=*/1,
                                    /*rank=*/parallel_args_.rank(),
                                    /*world_size=*/parallel_args_.world_size());
  if (qweight.defined()) {
    CHECK_EQ(qweight_.sizes(), qweight.sizes())
        << "qweight size mismatch for " << name();
    qweight_.copy_(qweight);
    qweight_is_loaded_ = true;
  }
  const auto qzeros =
      state_dict.get_sharded_tensor("qzeros",
                                    /*dim=*/1,
                                    /*rank=*/parallel_args_.rank(),
                                    /*world_size=*/parallel_args_.world_size());
  if (qzeros.defined()) {
    CHECK_EQ(qzeros_.sizes(), qzeros.sizes())
        << "qzeros size mismatch for " << name();
    qzeros_.copy_(qzeros);
    qzeros_is_loaded_ = true;
  }
  const auto scales =
      state_dict.get_sharded_tensor("scales",
                                    /*dim=*/1,
                                    /*rank=*/parallel_args_.rank(),
                                    /*world_size=*/parallel_args_.world_size());
  if (scales.defined()) {
    CHECK_EQ(scales_.sizes(), scales.sizes())
        << "scales size mismatch for " << name();
    scales_.copy_(scales);
    scales_is_loaded_ = true;
  }
  const auto g_idx = state_dict.get_tensor("g_idx");
  if (g_idx.defined()) {
    CHECK_EQ(g_idx_.sizes(), g_idx.sizes())
        << "g_idx size mismatch for " << name();
    g_idx_.copy_(g_idx);
    g_idx_is_loaded_ = true;
  }
}

// special load_state_dict for fused cases
void ColumnParallelQuantLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string_view>& prefixes) {
  if (qweight_list_.size() < prefixes.size()) {
    qweight_list_.resize(prefixes.size());
    qzeros_list_.resize(prefixes.size());
    scales_list_.resize(prefixes.size());
    g_idx_list_.resize(prefixes.size());
  }

  for (size_t i = 0; i < prefixes.size(); ++i) {
    std::string name = std::string(prefixes[i]) + "qweight";
    const auto qweight = state_dict.get_sharded_tensor(
        name,
        /*dim=*/1,
        /*rank=*/parallel_args_.rank(),
        /*world_size=*/parallel_args_.world_size());
    if (qweight.defined()) {
      CHECK(!qweight_list_[i].defined()) << "qweight already loaded";
      // make a copy in case the checkpoint is deleted
      qweight_list_[i] = qweight.clone();
    }
    name = std::string(prefixes[i]) + "qzeros";
    const auto qzeros = state_dict.get_sharded_tensor(
        name,
        /*dim=*/1,
        /*rank=*/parallel_args_.rank(),
        /*world_size=*/parallel_args_.world_size());
    if (qzeros.defined()) {
      CHECK(!qzeros_list_[i].defined()) << "qzeros already loaded";
      // make a copy in case the checkpoint is deleted
      qzeros_list_[i] = qzeros.clone();
    }
    name = std::string(prefixes[i]) + "scales";
    const auto scales = state_dict.get_sharded_tensor(
        name,
        /*dim=*/1,
        /*rank=*/parallel_args_.rank(),
        /*world_size=*/parallel_args_.world_size());
    if (scales.defined()) {
      CHECK(!scales_list_[i].defined()) << "scales already loaded";
      // make a copy in case the checkpoint is deleted
      scales_list_[i] = scales.clone();
    }
    name = std::string(prefixes[i]) + "g_idx";
    const auto g_idx = state_dict.get_tensor(name);
    if (g_idx.defined()) {
      CHECK(!g_idx_list_[i].defined()) << "g_idx already loaded";
      // make a copy in case the checkpoint is deleted
      g_idx_list_[i] = g_idx.clone();
    }
  }
}

void ColumnParallelQuantLinearImpl::verify_loaded_weights() {
  if (qweight_list_.empty()) {
    CHECK(qweight_is_loaded_) << "qweight is not loaded for " << name();
    CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << name();
    CHECK(scales_is_loaded_) << "scales is not loaded for " << name();
    CHECK(g_idx_is_loaded_) << "g_idx is not loaded for " << name();
    return;
  }
  auto qweight = torch::cat(qweight_list_, /*dim=*/1);
  qweight_list_.clear();
  CHECK_EQ(qweight_.sizes(), qweight.sizes())
      << "qweight size mismatch for " << name();
  qweight_.copy_(qweight);

  auto qzeros = torch::cat(qzeros_list_, /*dim=*/1);
  qzeros_list_.clear();
  CHECK_EQ(qzeros_.sizes(), qzeros.sizes())
      << "qzeros size mismatch for " << name();
  qzeros_.copy_(qzeros);

  auto scales = torch::cat(scales_list_, /*dim=*/1);
  scales_list_.clear();
  CHECK_EQ(scales_.sizes(), scales.sizes())
      << "scales size mismatch for " << name();
  scales_.copy_(scales);

  // all g_idx should be the same cross all shards, use the first one
  CHECK_EQ(g_idx_.sizes(), g_idx_list_[0].sizes())
      << "g_idx size mismatch for " << name();
  g_idx_.copy_(g_idx_list_[0]);
  g_idx_list_.clear();
}

RowParallelQuantLinearImpl::RowParallelQuantLinearImpl(
    int64_t in_features,
    int64_t out_features,
    int64_t bits,
    int64_t group_size,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    const torch::ScalarType& /*dtype*/,
    const torch::Device& device)
    : in_features_(in_features),
      out_features_(out_features),
      bits_(bits),
      group_size_(group_size),
      parallel_args_(parallel_args),
      input_is_parallelized_(input_is_parallelized) {
  CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 8)
      << "Only 2,3,4,8 bits are supported";
  CHECK(group_size > 0) << "group_size must be positive";
  CHECK(in_features % 32 == 0) << "in_features must be divisible by 32";
  CHECK(in_features % group_size == 0) << "in_features must be divisible by "
                                       << group_size;
  CHECK(out_features % 32 == 0) << "out_features must be divisible by 32";

  const auto world_size = parallel_args_.world_size();
  CHECK(in_features % world_size == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size;
  const int64_t in_features_per_partition = in_features / world_size;

  // TODO: support partioning
  qweight_ = register_parameter(
      "qweight",
      torch::empty({in_features_per_partition / 32 * bits, out_features},
                   torch::dtype(torch::kInt32).device(device)),
      /*requires_grad=*/false);
  qzeros_ = register_parameter(
      "qzeros",
      torch::empty({round_up(in_features_per_partition, group_size),
                    out_features / 32 * bits},
                   torch::dtype(torch::kInt32).device(device)),
      /*requires_grad=*/false);

  scales_ = register_parameter(
      "scales",
      torch::empty(
          {round_up(in_features_per_partition, group_size), out_features},
          torch::dtype(torch::kFloat16).device(device)),
      /*requires_grad=*/false);

  std::vector<int> g_idx_data;
  for (int64_t i = 0; i < in_features; ++i) {
    g_idx_data.push_back(i / group_size);
  }
  g_idx_ = register_buffer(
      "g_idx",
      torch::tensor(g_idx_data, torch::dtype(torch::kInt32).device(device)));
}

torch::Tensor RowParallelQuantLinearImpl::forward(torch::Tensor input) {
  if (!input_is_parallelized_) {
    input = scatter_to_model_parallel_region(input, parallel_args_);
  }
  auto output =
      torch::empty({input.size(0), out_features_},
                   torch::dtype(torch::kFloat32).device(input.device()));
  vec_quant_matmul_cuda(
      input, qweight_, output, scales_, qzeros_, g_idx_, bits_);
  if (parallel_args_.world_size() > 1) {
    output = reduce_from_model_parallel_region(output, parallel_args_);
  }
  return output.to(input);
}

// load the weight from the checkpoint
void RowParallelQuantLinearImpl::load_state_dict(const StateDict& state_dict) {
  const auto qweight =
      state_dict.get_sharded_tensor("qweight",
                                    /*dim=*/0,
                                    /*rank=*/parallel_args_.rank(),
                                    /*world_size=*/parallel_args_.world_size());
  if (qweight.defined()) {
    CHECK_EQ(qweight_.sizes(), qweight.sizes())
        << "qweight size mismatch for " << name();
    qweight_.copy_(qweight);
    qweight_is_loaded_ = true;
  }
  const auto qzeros =
      state_dict.get_sharded_tensor("qzeros",
                                    /*dim=*/0,
                                    /*rank=*/parallel_args_.rank(),
                                    /*world_size=*/parallel_args_.world_size());
  if (qzeros.defined()) {
    CHECK_EQ(qzeros_.sizes(), qzeros.sizes())
        << "qzeros size mismatch for " << name();
    qzeros_.copy_(qzeros);
    qzeros_is_loaded_ = true;
  }
  const auto scales =
      state_dict.get_sharded_tensor("scales",
                                    /*dim=*/0,
                                    /*rank=*/parallel_args_.rank(),
                                    /*world_size=*/parallel_args_.world_size());
  if (scales.defined()) {
    CHECK_EQ(scales_.sizes(), scales.sizes())
        << "scales size mismatch for " << name();
    scales_.copy_(scales);
    scales_is_loaded_ = true;
  }
  const auto g_idx = state_dict.get_tensor("g_idx");
  if (g_idx.defined()) {
    CHECK_EQ(g_idx_.sizes(), g_idx.sizes())
        << "g_idx size mismatch for " << name();
    g_idx_.copy_(g_idx);
    g_idx_is_loaded_ = true;
  }
}

void RowParallelQuantLinearImpl::verify_loaded_weights() const {
  CHECK(qweight_is_loaded_) << "qweight is not loaded for " << name();
  CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << name();
  CHECK(scales_is_loaded_) << "scales is not loaded for " << name();
  CHECK(g_idx_is_loaded_) << "g_idx is not loaded for " << name();
}

}  // namespace llm
