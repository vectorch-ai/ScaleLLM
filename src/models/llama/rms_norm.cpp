#include "rms_norm.h"
#include <c10/core/ScalarType.h>

namespace llm {

RMSNormImpl::RMSNormImpl(int64_t dim, float eps) : eps_(eps) {
  weight_ = register_parameter("weight", torch::empty({dim}));
}

torch::Tensor RMSNormImpl::norm(torch::Tensor x) {
  return x * torch::rsqrt(x.pow(2).mean(-1, /*keepdim*/true) + eps_);
}

torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
  auto output = norm(x.toType(torch::kFloat)).type_as(x);
  return output * weight_;
}

// load the weight from the checkpoint
void RMSNormImpl::load_state_dict(const StateDict& state_dict) {
  weight_.copy_(state_dict.get_tensor("weight"));
}

}  // namespace llm
