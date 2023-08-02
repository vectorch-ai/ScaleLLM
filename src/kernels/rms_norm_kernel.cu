// copied from vllm for build testing purpose
// https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
// TODO: remove this file after build testing

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "reduce_kernel_utils.cuh"

namespace llm::kernel {

// calculate the root mean square norm.
// equation: out = (x / sqrt(sum(x^2) / dim)) * weight
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ out,           // [num_tokens, dim]
    const scalar_t* __restrict__ input,   // [num_tokens, dim]
    const scalar_t* __restrict__ weight,  // [dim]
    const float epsilon,
    const int dim) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * dim + idx];
    variance += x * x;
  }
  variance = block_reduce_sum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / dim + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < dim; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * dim + idx];
    out[blockIdx.x * dim + idx] = ((scalar_t)(x * s_variance)) * weight[idx];
  }
}

void rms_norm(torch::Tensor& out,     // [num_tokens, dim]
              torch::Tensor& input,   // [num_tokens, dim]
              torch::Tensor& weight,  // [dim]
              float epsilon) {
  const int num_tokens = input.size(0);
  const int dim = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(dim, 1024));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "rms_norm_kernel",
      [&] {
        rms_norm_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                epsilon,
                dim);
      });
}

}  // namespace llm::kernel
