#pragma once

#include <ATen/core/TensorBody.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "kernels/layernorm_kernels.h"
#include "layers/module/module.h"
#include "layers/module/module_holder.h"
#include "model_loader/state_dict.h"

DECLARE_bool(disable_custom_kernels);
namespace llm {
namespace detail {

inline torch::Tensor norm(const torch::Tensor& x, float eps) {
  const auto mean = x.pow(/*exponent=*/2).mean(/*dim=*/-1, /*keepdim=*/true);
  return x * torch::rsqrt(mean + eps);
}

inline torch::Tensor rms_norm(const torch::Tensor& input,
                              const torch::Tensor& weight,
                              float eps) {
  // it is important to use float to calculate the mean and std
  const auto output = norm(input.to(torch::kFloat), eps);
  // convert back to the original dtype
  return output.to(input) * weight;
}

inline torch::Tensor gemma_rms_norm(const torch::Tensor& input,
                                    const torch::Tensor& weight,
                                    float eps) {
  // it is important to use float to calculate the mean and std
  auto output = norm(input.to(torch::kFloat), eps);
  // Llama does x.to(float16) * w whilst Gemma is (x * (w + 1)).to(float16)
  // See https://github.com/huggingface/transformers/pull/29402
  output = output * (1.0 + weight.to(torch::kFloat));
  return output.type_as(input);
}

inline torch::Tensor rms_norm_residual(const torch::Tensor& input,
                                       torch::Tensor& residual,
                                       const torch::Tensor& weight,
                                       float eps) {
  // it is important to use float for the residual
  auto x = input.to(torch::kFloat) + residual.to(torch::kFloat);
  residual = x.to(input);
  const auto output = norm(x, eps);
  // convert back to the original dtype
  return output.to(input) * weight;
}

inline torch::Tensor layer_norm(torch::Tensor input,
                                const std::vector<int64_t>& normalized_shape,
                                const torch::Tensor& weight,
                                const torch::Tensor& bias,
                                double eps) {
  namespace F = torch::nn::functional;
  return F::detail::layer_norm(input, normalized_shape, weight, bias, eps);
}

}  // namespace detail

// apply layer normalization over a mini-batch of inputs as described in
// the paper `Layer Normalization`: https://arxiv.org/abs/1607.06450
// x = ((x - mean(x)) / sqrt(std(x) + eps)) * weight + bias
class LayerNormImpl : public Module {
 public:
  // dim: the dim over which the mean and std are calculated separately.
  // eps: a value added to the denominator for numerical stability.
  LayerNormImpl(int64_t dim,
                float eps,
                bool bias,
                const torch::TensorOptions& options)
      : eps_(eps) {
    normalized_shape_ = {dim};
    weight_ =
        register_parameter("weight", torch::empty(normalized_shape_, options));
    if (bias) {
      bias_ =
          register_parameter("bias", torch::zeros(normalized_shape_, options));
    }
  }

  torch::Tensor forward(torch::Tensor input) {
    if (input.is_cuda() && !FLAGS_disable_custom_kernels) {
      auto output = torch::empty_like(input);
      kernel::layer_norm(output, input, weight_, bias_, eps_);
      return output;
    }
    namespace F = torch::nn::functional;
    return F::detail::layer_norm(
        input, normalized_shape_, weight_, bias_, eps_);
  }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  torch::Tensor bias_{nullptr};

  // whether the weight is loaded
  bool weight_is_loaded_ = false;
  bool bias_is_loaded_ = false;

  // configs
  float eps_;
  std::vector<int64_t> normalized_shape_;
};
LLM_MODULE(LayerNorm);

// Root mean square normalization
class RMSNormImpl : public Module {
 public:
  RMSNormImpl(int64_t dim, float eps, const torch::TensorOptions& options)
      : eps_(eps) {
    weight_ = register_parameter("weight", torch::empty({dim}, options));
  }

  torch::Tensor forward(const torch::Tensor& input) {
    if (input.is_cuda() && !FLAGS_disable_custom_kernels) {
      auto output = torch::empty_like(input);
      kernel::rms_norm(output, input, weight_, eps_);
      return output;
    }
    return detail::rms_norm(input, weight_, eps_);
  }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;

  // configs
  float eps_;
};
LLM_MODULE(RMSNorm);

class GemmaRMSNormImpl : public Module {
 public:
  GemmaRMSNormImpl(int64_t dim, float eps, const torch::TensorOptions& options)
      : eps_(eps) {
    weight_ = register_parameter("weight", torch::empty({dim}, options));
  }

  torch::Tensor forward(const torch::Tensor& input) {
    if (input.is_cuda() && !FLAGS_disable_custom_kernels) {
      auto output = torch::empty_like(input);
      kernel::gemma_rms_norm(output, input, weight_, eps_);
      return output;
    }
    return detail::gemma_rms_norm(input, weight_, eps_);
  }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;

  // configs
  float eps_;
};
LLM_MODULE(GemmaRMSNorm);

// Root mean square normalization
class RMSNormResidualImpl : public Module {
 public:
  RMSNormResidualImpl(int64_t dim,
                      float eps,
                      const torch::TensorOptions& options)
      : eps_(eps) {
    weight_ = register_parameter("weight", torch::empty({dim}, options));
  }

  torch::Tensor forward(const torch::Tensor& input, torch::Tensor& residual) {
    if (input.is_cuda() && !FLAGS_disable_custom_kernels) {
      auto output = torch::empty_like(input);
      if (residual.defined()) {
        kernel::rms_norm_residual(output, residual, input, weight_, eps_);
      } else {
        residual = input;
        kernel::rms_norm(output, input, weight_, eps_);
      }
      return output;
    }

    if (residual.defined()) {
      return detail::rms_norm_residual(input, residual, weight_, eps_);
    }
    residual = input;
    return detail::rms_norm(input, weight_, eps_);
  }

 private:
  // parameter members, must be registered
  torch::Tensor weight_{nullptr};

  // whether the weight is loaded
  bool is_loaded_ = false;

  // configs
  float eps_;
};
LLM_MODULE(RMSNormResidual);

}  // namespace llm
