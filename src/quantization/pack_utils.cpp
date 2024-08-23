#include "pack_utils.h"

#include <glog/logging.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/torch.h>

#include "common/array.h"

namespace llm::pack_utils {

// returns int32 packed qweight: (m, n) -> (m, n/pack_factor)
torch::Tensor pack_cols(const torch::Tensor& qweight,  // (m, n)
                        int64_t num_bits) {
  CHECK_EQ(qweight.dim(), 2);
  CHECK_EQ(qweight.scalar_type(), torch::kInt32);
  CHECK(qweight.is_contiguous());

  const int64_t size_m = qweight.size(0);
  const int64_t size_n = qweight.size(1);
  const int64_t pack_factor = 32 / num_bits;
  CHECK_EQ(size_n % pack_factor, 0);

  torch::Tensor qweight_cpu = qweight.cpu().contiguous();
  const Array qw(qweight_cpu.data_ptr<int32_t>(), make_shape(size_m, size_n));

  torch::Tensor packed_qweight =
      torch::empty({size_m, size_n / pack_factor},
                   torch::dtype(torch::kInt32).device(torch::kCPU));
  Array p_qw(packed_qweight.data_ptr<int32_t>(),
             make_shape(size_m, size_n / pack_factor));

  const int32_t mask = (1 << num_bits) - 1;
  for (int64_t i = 0; i < size_m; ++i) {
    for (int64_t j = 0; j < size_n; j += pack_factor) {
      int32_t packed_val = 0;
      for (int64_t k = 0; k < pack_factor; ++k) {
        packed_val |= (qw(i, j + k) & mask) << (num_bits * k);
      }
      p_qw(i, j / pack_factor) = packed_val;
    }
  }
  return packed_qweight.to(qweight);
}

// returns int32 qweight: (m, n/pack_factor) -> (m, n)
torch::Tensor unpack_cols(
    const torch::Tensor& packed_qweight,  // (m, n/pack_factor)
    int64_t num_bits) {
  CHECK_EQ(packed_qweight.dim(), 2);
  CHECK_EQ(packed_qweight.scalar_type(), torch::kInt32);
  CHECK(packed_qweight.is_contiguous());

  const int64_t pack_factor = 32 / num_bits;

  const int64_t size_m = packed_qweight.size(0);
  const int64_t size_n = packed_qweight.size(1) * pack_factor;

  torch::Tensor packed_qweight_cpu = packed_qweight.cpu().contiguous();
  const Array p_qw(packed_qweight_cpu.data_ptr<int32_t>(),
                   make_shape(size_m, size_n / pack_factor));

  torch::Tensor qweight = torch::empty(
      {size_m, size_n}, torch::dtype(torch::kInt32).device(torch::kCPU));
  Array qw(qweight.data_ptr<int32_t>(), make_shape(size_m, size_n));

  const int32_t mask = (1 << num_bits) - 1;
  for (int64_t i = 0; i < size_m; ++i) {
    for (int64_t j = 0; j < size_n; j += pack_factor) {
      const int32_t packed_val = p_qw(i, j / pack_factor);
      for (int64_t k = 0; k < pack_factor; ++k) {
        qw(i, j + k) = (packed_val >> (num_bits * k)) & mask;
      }
    }
  }
  return qweight.to(packed_qweight);
}

}  // namespace llm::pack_utils