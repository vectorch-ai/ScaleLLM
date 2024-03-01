#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include "dispatch.h"
#include "reduce_kernel_utils.cuh"

namespace llm::kernel {

// calculate the root mean square norm.
// equation: x -> w * x / sqrt(E[x^2] + eps)
// The mean is calculated over the last dimension
// equilvalent to layernorm module in the T5 style No bias and no subtraction of
// mean.
template <typename T>
__global__ void rms_norm_kernel(T* __restrict__ out,
                                const T* __restrict__ input,
                                const T* __restrict__ weight,
                                const float epsilon,
                                int n) {
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  __shared__ float s_variance;
  float variance = 0.0f;

  for (int i = tidx; i < n; i += blockDim.x) {
    const float x = __ldg(&input[bidx * n + i]);
    variance += x * x;
  }
  variance = block_reduce_sum<float>(variance);
  if (tidx == 0) {
    s_variance = rsqrtf(variance / n + epsilon);
  }
  __syncthreads();

  for (int i = tidx; i < n; i += blockDim.x) {
    const int idx = bidx * n + i;
    const float x = __ldg(&input[idx]);
    out[idx] = (T)(x * s_variance * weight[i]);
  }
}

void rms_norm(torch::Tensor& out,
              torch::Tensor input,
              torch::Tensor weight,
              float epsilon) {
  DCHECK(input.is_contiguous()) << "input tensor must be contiguous";
  DCHECK(out.is_contiguous()) << "output tensor must be contiguous";

  const int n = input.size(1);

  dim3 grid(input.size(0));
  dim3 block(std::min(n, 1024));
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    rms_norm_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            epsilon,
            n);
  });
}

// equation: x -> (x - E[x]) / sqrt(Var[x] + eps) * w + b
// The mean and standard-deviation are calculated over the last dimension
template <typename T>
__global__ void layer_norm_kernel(T* __restrict__ out,
                                  const T* __restrict__ input,
                                  const T* __restrict__ weight,
                                  const T* __restrict__ bias,
                                  const float epsilon,
                                  int n) {
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  // calculate mean of the input.
  for (int i = tidx; i < n; i += blockDim.x) {
    const int idx = bidx * n + i;
    mean += __ldg(&input[idx]);
  }
  mean = block_reduce_sum<float>(mean);
  if (tidx == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  // calculate variance of the input.
  for (int i = tidx; i < n; i += blockDim.x) {
    const float x = input[bidx * n + i] - s_mean;
    variance += x * x;
  }
  variance = block_reduce_sum<float>(variance);
  if (tidx == 0) {
    s_variance = rsqrtf(variance / n + epsilon);
  }
  __syncthreads();

  for (int i = tidx; i < n; i += blockDim.x) {
    const int idx = bidx * n + i;
    float local_out =
        (__ldg(&input[idx]) - s_mean) * s_variance * __ldg(&weight[i]);
    if (bias != nullptr) {
      local_out += __ldg(&bias[i]);
    }
    out[idx] = (T)(local_out);
  }
}

void layer_norm(torch::Tensor& out,
                torch::Tensor input,
                torch::Tensor weight,
                torch::Tensor bias,
                float epsilon) {
  DCHECK(input.is_contiguous()) << "input tensor must be contiguous";
  DCHECK(out.is_contiguous()) << "output tensor must be contiguous";

  const int n = input.size(1);

  dim3 grid(input.size(0));
  dim3 block(std::min(n, 1024));
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "layer_norm_kernel", [&] {
    layer_norm_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.defined() ? bias.data_ptr<scalar_t>() : nullptr,
            epsilon,
            n);
  });
}

}  // namespace llm::kernel
