#include "qlinear_exllamav2_impl.h"

#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch/types.h>

#include "model_loader/state_dict.h"
#include "model_parallel/model_parallel.h"
#include "models/model_args.h"

extern uintptr_t make_q_matrix(torch::Tensor q_weight,
                               torch::Tensor q_perm,
                               torch::Tensor q_invperm,
                               torch::Tensor q_scale,
                               torch::Tensor q_scale_max,
                               torch::Tensor q_groups,
                               torch::Tensor gptq_qzeros,
                               torch::Tensor gptq_scales,
                               torch::Tensor gptq_g_idx,
                               torch::Tensor temp_dq);

extern void gemm_half_q_half(torch::Tensor a,
                             uintptr_t b,
                             torch::Tensor c,
                             bool force_cuda);

extern void free_q_matrix(uintptr_t w);

namespace llm {

namespace {
const auto none_tensor = torch::empty({1, 1}, torch::kMeta);

// we use thread_local to reuse the buffer for quant_matmul
// with a assumption that each thread will only operate on single device and
// model initialization and forward will be on the same thread.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local torch::Tensor tl_temp_dq;

void allocate_temp_dq(int64_t size, const torch::Device& device) {
  if (tl_temp_dq.defined()) {
    CHECK(tl_temp_dq.device() == device)
        << "temp_dq was allocated on " << tl_temp_dq.device()
        << " but now is on " << device;
  }

  if (!tl_temp_dq.defined() || tl_temp_dq.numel() < size) {
    // reallocate tl_temp_dq if it is not defined or not large enough
    tl_temp_dq =
        torch::empty({size}, torch::dtype(torch::kHalf).device(device));
  }
}
}  // namespace

ColumnParallelQLinearExllamav2Impl::ColumnParallelQLinearExllamav2Impl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : ColumnParallelQLinearImpl(in_features,
                                out_features,
                                bias,
                                quant_args,
                                /*qweight_pack_dim=*/0,
                                gather_output,
                                parallel_args,
                                options) {
  const auto bits = quant_args.bits();
  CHECK(bits == 4) << "Only 4 bits are supported";

  const auto group_size =
      quant_args.group_size() > 0 ? quant_args.group_size() : in_features;

  if (quant_args.desc_act()) {
    std::vector<int32_t> g_idx_data;
    g_idx_data.reserve(in_features);
    for (int32_t i = 0; i < in_features; ++i) {
      g_idx_data.push_back(i / group_size);
    }
    // TODO: load g_idx from weights when desc_act is true
    // g_idx has to be on cpu in the ac-order case, otherwise the q_matrix_ will
    // be segfault
    g_idx_ = register_buffer(
        "g_idx",
        torch::tensor(g_idx_data,
                      torch::dtype(torch::kInt32).device(torch::kCPU)));
    q_perm_ = register_buffer(
        "q_perm", torch::empty({in_features}, options.dtype(torch::kShort)));
    q_invperm_ = register_buffer(
        "q_invperm", torch::empty({in_features}, options.dtype(torch::kShort)));
  } else {
    g_idx_ = none_tensor;
    q_perm_ = none_tensor;
    q_invperm_ = none_tensor;
  }

  const int64_t temp_dq_size = in_features * out_features * 2 + 128;
  allocate_temp_dq(temp_dq_size, options.device());
}

ColumnParallelQLinearExllamav2Impl::~ColumnParallelQLinearExllamav2Impl() {
  if (q_matrix_ != 0) {
    free_q_matrix(q_matrix_);
  }
}

torch::Tensor ColumnParallelQLinearExllamav2Impl::quant_matmul(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales) const {
  // lazy initialization
  if (q_matrix_ == 0) {
    CHECK(tl_temp_dq.defined()) << "tl_temp_dq is not defined. model "
                                   "initialization and forward should be on "
                                   "the same thread";
    q_matrix_ = make_q_matrix(qweight,
                              q_perm_,
                              q_invperm_,
                              none_tensor,
                              none_tensor,
                              none_tensor,
                              qzeros,
                              scales,
                              g_idx_,
                              tl_temp_dq);
  }
  const int64_t out_features = qweight.size(-1);
  torch::Tensor output =
      torch::empty({input.size(0), out_features}, input.options());
  gemm_half_q_half(input, q_matrix_, output, /*force_cuda=*/false);
  return output;
}

RowParallelQLinearExllamav2Impl::RowParallelQLinearExllamav2Impl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    bool input_is_parallelized,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : RowParallelQLinearImpl(in_features,
                             out_features,
                             bias,
                             quant_args,
                             /*qweight_pack_dim=*/0,
                             input_is_parallelized,
                             parallel_args,
                             options) {
  const auto bits = quant_args.bits();
  CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 8)
      << "Only 2,3,4,8 bits are supported";

  const auto group_size =
      quant_args.group_size() > 0 ? quant_args.group_size() : in_features;

  if (quant_args.desc_act()) {
    std::vector<int32_t> g_idx_data;
    g_idx_data.reserve(in_features);
    for (int32_t i = 0; i < in_features; ++i) {
      g_idx_data.push_back(i / group_size);
    }
    // TODO: load g_idx from weights when desc_act is true
    // g_idx has to be on cpu in the ac-order case, otherwise the q_matrix_ will
    // be segfault
    g_idx_ = register_buffer(
        "g_idx",
        torch::tensor(g_idx_data,
                      torch::dtype(torch::kInt32).device(torch::kCPU)));
    q_perm_ = register_buffer(
        "q_perm", torch::empty({in_features}, options.dtype(torch::kShort)));
    q_invperm_ = register_buffer(
        "q_invperm", torch::empty({in_features}, options.dtype(torch::kShort)));
  } else {
    g_idx_ = none_tensor;
    q_perm_ = none_tensor;
    q_invperm_ = none_tensor;
  }

  const int64_t temp_dq_size = in_features * out_features * 2 + 128;
  allocate_temp_dq(temp_dq_size, options.device());
}

RowParallelQLinearExllamav2Impl::~RowParallelQLinearExllamav2Impl() {
  if (q_matrix_ != 0) {
    free_q_matrix(q_matrix_);
  }
}

torch::Tensor RowParallelQLinearExllamav2Impl::quant_matmul(
    const torch::Tensor& input,
    const torch::Tensor& qweight,
    const torch::Tensor& qzeros,
    const torch::Tensor& scales) const {
  if (q_matrix_ == 0) {
    CHECK(tl_temp_dq.defined()) << "tl_temp_dq is not defined. model "
                                   "initialization and forward should be on "
                                   "the same thread";
    q_matrix_ = make_q_matrix(qweight,
                              q_perm_,
                              q_invperm_,
                              none_tensor,
                              none_tensor,
                              none_tensor,
                              qzeros,
                              scales,
                              g_idx_,
                              tl_temp_dq);
  }
  const int64_t out_features = qweight.size(-1);
  torch::Tensor output =
      torch::empty({input.size(0), out_features}, input.options());
  gemm_half_q_half(input, q_matrix_, output, /*force_cuda=*/false);
  return output;
}

}  // namespace llm
