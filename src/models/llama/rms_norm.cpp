#include "rms_norm.h"

#include <c10/core/ScalarType.h>
#include <glog/logging.h>

namespace llm {

RMSNormImpl::RMSNormImpl(int64_t dim, float eps) : eps_(eps) {
  weight_ = register_parameter(
      "weight", torch::empty({dim}), /*requires_grad=*/false);
}

torch::Tensor RMSNormImpl::norm(torch::Tensor x) {
  return x * torch::rsqrt(x.pow(2).mean(-1, /*keepdim*/ true) + eps_);
}

torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
  auto output = norm(x.to(torch::kFloat)).type_as(x);
  return output * weight_;
}

// load the weight from the checkpoint
void RMSNormImpl::load_state_dict(const StateDict& state_dict) {
  const auto weight = state_dict.get_tensor("weight");
  if (weight.defined()) {
    CHECK_EQ(weight_.sizes(), weight.sizes()) << "weight size mismatch";
    weight_.copy_(weight);
  } else {
    LOG(WARNING) << "weight is not defined";
  }
}

}  // namespace llm
