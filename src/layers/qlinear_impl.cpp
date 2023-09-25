#include "qlinear_impl.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "model_loader/state_dict.h"
#include "model_parallel.h"
#include "models/args.h"

DEFINE_string(qlinear_gptq_impl,
              "",
              "type of qlinear gptq impl, slow, cuda, or empty for auto");

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
  if (FLAGS_qlinear_gptq_impl == "slow") {
    auto weights = details::construct_weights(mat,
                                              zeros,
                                              scales,
                                              g_idx,
                                              /*bits=*/bits);
    torch::matmul_out(/*out=*/mul, /*self=*/vec, /*other=*/weights);
    return;
  }

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

int64_t round_up(int64_t num, int64_t multiple) {
  return ((num + multiple - 1) / multiple);
}

}  // namespace

namespace details {
// construct weights matrix for gptq from quantized weights
// return the weights matrix [in_features, out_features] with following formula:
// weights = scales * (qweights - qzeros)
torch::Tensor construct_weights(
    const torch::Tensor& qweights,  // [n_ints, out_features] IntTensor
    const torch::Tensor& qzeros,    // [n_groups, n_ints] IntTensor
    const torch::Tensor& scales,    // [n_groups, out_features] HalfTensor
    const torch::Tensor& g_idx,     // [in_features] IntTensor
    int64_t bits) {
  CHECK(bits == 2 || bits == 4 || bits == 8) << "Only 2,4,8 bits are supported";

  std::vector<int32_t> bits_to_shift;
  for (int32_t i = 0; i < 32; i += bits) {
    bits_to_shift.push_back(i);
  }
  // [1, 32/bits]
  auto shift_bits =
      torch::tensor(bits_to_shift, qweights.options()).unsqueeze(0);
  auto dtype = (bits == 8) ? torch::kInt16 : torch::kInt8;
  // [n_groups, out_features/n_bits, n_ints]
  auto zeros = torch::bitwise_right_shift(
                   qzeros.unsqueeze(2).expand({-1, -1, 32 / bits}),
                   shift_bits.unsqueeze(0))
                   .to(dtype);
  zeros.bitwise_and_((1 << bits) - 1);
  zeros.add_(1);
  // [n_groups, out_features]
  zeros = zeros.reshape(scales.sizes());

  auto weights = torch::bitwise_right_shift(
                     qweights.unsqueeze(1).expand({-1, 32 / bits, -1}),
                     shift_bits.unsqueeze(-1))
                     .to(dtype);
  weights.bitwise_and_((1 << bits) - 1);
  weights = weights.reshape({-1, qweights.size(1)});
  // auto gathered_scales = scales.gather(/*dim=*/0, /*index=*/g_idx);
  // auto gathered_zeros = zeros.gather(/*dim=*/0, /*index=*/g_idx);
  return scales.index({g_idx}) * (weights - zeros.index({g_idx}));
}

// construct weights matrix for gptq from quantized weights without using g_idx
// slower than construct_weights with g_idx
// return the weights matrix [in_features, out_features] with following formula:
// weights = scales * (qweights - qzeros)
torch::Tensor construct_weights(
    const torch::Tensor& qweights,  // [n_ints, out_features] IntTensor
    const torch::Tensor& qzeros,    // [n_groups, n_ints] IntTensor
    const torch::Tensor& scales,    // [n_groups, out_features] HalfTensor
    int64_t bits) {
  CHECK(bits == 2 || bits == 4 || bits == 8) << "Only 2,4,8 bits are supported";

  std::vector<int32_t> bits_to_shift;
  for (int32_t i = 0; i < 32; i += bits) {
    bits_to_shift.push_back(i);
  }

  // [1, n_ints=32/bits]
  auto shift_bits =
      torch::tensor(bits_to_shift, qweights.options()).unsqueeze(0);
  auto dtype = (bits == 8) ? torch::kInt16 : torch::kInt8;
  // [n_groups, out_features/n_bits, n_ints]
  auto zeros = torch::bitwise_right_shift(
                   qzeros.unsqueeze(2).expand({-1, -1, 32 / bits}),
                   shift_bits.unsqueeze(0))
                   .to(dtype);
  zeros.bitwise_and_((1 << bits) - 1);
  zeros.add_(1);
  // [n_groups, 1, out_features]
  zeros = zeros.reshape({scales.size(0), 1, scales.size(1)});

  auto weights = torch::bitwise_right_shift(
                     qweights.unsqueeze(1).expand({-1, 32 / bits, -1}),
                     shift_bits.unsqueeze(-1))
                     .to(dtype);
  weights.bitwise_and_((1 << bits) - 1);
  // [n_groups, group_size, out_features]
  weights = weights.reshape({scales.size(0), -1, scales.size(1)});
  weights = scales.unsqueeze(1) * (weights - zeros);
  return weights.reshape({-1, scales.size(1)});
}
}  // namespace details

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
  CHECK(in_features % group_size == 0)
      << "in_features must be divisible by " << group_size;
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

torch::Tensor ColumnParallelQuantLinearImpl::forward(
    torch::Tensor input) const {
  auto input_float = input.to(torch::kFloat32);
  auto scales_float = scales_.to(torch::kFloat32);
  auto output_float =
      torch::empty({input_float.size(0), out_features_}, input_float.options());

  vec_quant_matmul_cuda(input_float,
                        qweight_,
                        output_float,
                        scales_float,
                        qzeros_,
                        g_idx_,
                        bits_);

  auto output = output_float.to(input);
  if (parallel_args_.world_size() > 1 && gather_output_) {
    output = gather_from_model_parallel_region(output, parallel_args_);
  }
  return output;
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
}

bool ColumnParallelQuantLinearImpl::load_weights(
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
void ColumnParallelQuantLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string_view>& prefixes) {
  if (qweight_list_.size() < prefixes.size()) {
    qweight_list_.resize(prefixes.size());
    qzeros_list_.resize(prefixes.size());
    scales_list_.resize(prefixes.size());
  }

  for (size_t i = 0; i < prefixes.size(); ++i) {
    std::string tensor_name = std::string(prefixes[i]) + "qweight";
    const auto qweight = state_dict.get_sharded_tensor(
        tensor_name,
        /*dim=*/1,
        /*rank=*/parallel_args_.rank(),
        /*world_size=*/parallel_args_.world_size());
    if (qweight.defined()) {
      CHECK(!qweight_list_[i].defined()) << "qweight already loaded";
      // make a copy in case the checkpoint is deleted
      qweight_list_[i] = qweight.clone();
    }
    tensor_name = std::string(prefixes[i]) + "qzeros";
    const auto qzeros = state_dict.get_sharded_tensor(
        tensor_name,
        /*dim=*/1,
        /*rank=*/parallel_args_.rank(),
        /*world_size=*/parallel_args_.world_size());
    if (qzeros.defined()) {
      CHECK(!qzeros_list_[i].defined()) << "qzeros already loaded";
      // make a copy in case the checkpoint is deleted
      qzeros_list_[i] = qzeros.clone();
    }
    tensor_name = std::string(prefixes[i]) + "scales";
    const auto scales = state_dict.get_sharded_tensor(
        tensor_name,
        /*dim=*/1,
        /*rank=*/parallel_args_.rank(),
        /*world_size=*/parallel_args_.world_size());
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

void ColumnParallelQuantLinearImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(qweight_is_loaded_)
      << "qweight is not loaded for " << prefix + ".qweight";
  CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << prefix + ".qzeros";
  CHECK(scales_is_loaded_) << "scales is not loaded for " << prefix + ".scales";
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
  CHECK(in_features % group_size == 0)
      << "in_features must be divisible by " << group_size;
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
          torch::dtype(torch::kFloat32).device(device)),
      /*requires_grad=*/false);

  std::vector<int> g_idx_data;
  for (int64_t i = 0; i < in_features; ++i) {
    g_idx_data.push_back(i / group_size);
  }
  g_idx_ = register_buffer(
      "g_idx",
      torch::tensor(g_idx_data, torch::dtype(torch::kInt32).device(device)));
}

torch::Tensor RowParallelQuantLinearImpl::forward(torch::Tensor input) const {
  if (!input_is_parallelized_) {
    input = scatter_to_model_parallel_region(input, parallel_args_);
  }

  auto input_float = input.to(torch::kFloat32);
  auto scales_float = scales_.to(torch::kFloat32);
  auto output_float =
      torch::empty({input_float.size(0), out_features_}, input_float.options());
  vec_quant_matmul_cuda(input_float,
                        qweight_,
                        output_float,
                        scales_float,
                        qzeros_,
                        g_idx_,
                        bits_);

  auto output = output_float.to(input);
  if (parallel_args_.world_size() > 1) {
    output = reduce_from_model_parallel_region(output, parallel_args_);
  }
  return output;
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
}

void RowParallelQuantLinearImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(qweight_is_loaded_)
      << "qweight is not loaded for " << prefix + ".qweight";
  CHECK(qzeros_is_loaded_) << "qzeros is not loaded for " << prefix + ".qzeros";
  CHECK(scales_is_loaded_) << "scales is not loaded for " << prefix + ".scales";
}

}  // namespace llm
