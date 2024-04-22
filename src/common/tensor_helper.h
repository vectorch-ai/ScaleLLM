#pragma once

#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <vector>

namespace llm {

template <typename T>
inline torch::Tensor create_2d_tensor(const std::vector<std::vector<T>>& vec,
                                      torch::ScalarType dtype) {
  if (vec.empty()) {
    return {};
  }

  const size_t n_rows = vec.size();
  const size_t n_cols = vec[0].size();
  auto tensor = torch::empty(
      {static_cast<int64_t>(n_rows), static_cast<int64_t>(n_cols)}, dtype);
  for (int64_t i = 0; i < n_rows; ++i) {
    CHECK_EQ(vec[i].size(), n_cols);
    tensor[i] = torch::tensor(vec[i], dtype);
  }
  return tensor;
};

inline torch::Tensor safe_to(const torch::Tensor& t,
                             const torch::Device& device) {
  return t.defined() ? t.to(device) : t;
};

inline torch::Tensor safe_to(const torch::Tensor& t,
                             const torch::TensorOptions& options) {
  return t.defined() ? t.to(options) : t;
};

}  // namespace llm