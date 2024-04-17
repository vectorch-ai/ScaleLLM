#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "dispatch.h"
#include "layernorm_kernels.h"
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
                                int64_t n) {
  const auto tidx = threadIdx.x;
  const auto bidx = blockIdx.x;

  __shared__ float s_variance;
  float variance = 0.0f;

  for (int64_t i = tidx; i < n; i += blockDim.x) {
    const float x = input[bidx * n + i];
    variance += x * x;
  }
  variance = block_reduce_sum<float>(variance);
  if (tidx == 0) {
    s_variance = rsqrtf(variance / n + epsilon);
  }
  __syncthreads();

  for (int64_t i = tidx; i < n; i += blockDim.x) {
    const int64_t idx = bidx * n + i;
    const float x = input[idx];
    out[idx] = (T)(x * s_variance) * weight[i];
  }
}

void rms_norm(torch::Tensor& out,
              torch::Tensor input,
              torch::Tensor weight,
              float epsilon) {
  DCHECK(input.is_contiguous()) << "input tensor must be contiguous";
  DCHECK(out.is_contiguous()) << "output tensor must be contiguous";

  const int64_t n = input.size(1);

  dim3 grid(input.size(0));
  dim3 block(std::min<int>(n, 1024));
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

// calculate the root mean square norm.
// equation: x -> w * x / sqrt(E[x^2] + eps)
// The mean is calculated over the last dimension
// equilvalent to layernorm module in the T5 style No bias and no subtraction of
// mean.
template <typename T>
__global__ void rms_norm_residual_kernel(T* __restrict__ out,
                                         T* __restrict__ residual,
                                         const T* __restrict__ input,
                                         const T* __restrict__ weight,
                                         const float epsilon,
                                         int64_t n) {
  const auto tidx = threadIdx.x;
  const auto bidx = blockIdx.x;

  __shared__ float s_variance;
  float variance = 0.0f;

  for (int64_t i = tidx; i < n; i += blockDim.x) {
    const int64_t idx = bidx * n + i;
    const float r = residual[idx];
    const float x = r + input[idx];
    residual[idx] = x;
    variance += x * x;
  }
  variance = block_reduce_sum<float>(variance);
  if (tidx == 0) {
    s_variance = rsqrtf(variance / n + epsilon);
  }
  __syncthreads();

  for (int64_t i = tidx; i < n; i += blockDim.x) {
    const int64_t idx = bidx * n + i;
    const float x = residual[idx];
    out[idx] = (T)(x * s_variance) * weight[i];
  }
}

void rms_norm_residual(torch::Tensor& out,
                       torch::Tensor& residual,
                       torch::Tensor input,
                       torch::Tensor weight,
                       float epsilon) {
  DCHECK(input.is_contiguous()) << "input tensor must be contiguous";
  DCHECK(out.is_contiguous()) << "output tensor must be contiguous";
  DCHECK(residual.is_contiguous()) << "residual tensor must be contiguous";

  const int64_t n = input.size(1);

  dim3 grid(input.size(0));
  dim3 block(std::min<int>(n, 1024));
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_residual_kernel", [&] {
    rms_norm_residual_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(),
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
                                  int64_t n) {
  const auto tidx = threadIdx.x;
  const auto bidx = blockIdx.x;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float mean = 0.0f;
  float variance = 0.0f;

  // calculate mean of the input.
  for (int64_t i = tidx; i < n; i += blockDim.x) {
    mean += input[bidx * n + i];
  }
  mean = block_reduce_sum<float>(mean);
  if (tidx == 0) {
    s_mean = mean / n;
  }
  __syncthreads();

  // calculate variance of the input.
  for (int64_t i = tidx; i < n; i += blockDim.x) {
    const float x = input[bidx * n + i] - s_mean;
    variance += x * x;
  }
  variance = block_reduce_sum<float>(variance);
  if (tidx == 0) {
    s_variance = rsqrtf(variance / n + epsilon);
  }
  __syncthreads();

  for (int64_t i = tidx; i < n; i += blockDim.x) {
    const int64_t idx = bidx * n + i;
    float local_out = (input[idx] - s_mean) * s_variance * weight[i];
    if (bias != nullptr) {
      local_out += bias[i];
    }
    out[idx] = (T)(local_out);
  }
}

// equation: x -> (x - E[x]) / sqrt(Var[x] + eps) * w + b
// The mean and standard-deviation are calculated over the last dimension
template <>
__global__ void layer_norm_kernel<half2>(half2* __restrict__ out,
                                         const half2* __restrict__ input,
                                         const half2* __restrict__ weight,
                                         const half2* __restrict__ bias,
                                         const float epsilon,
                                         int n) {
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;

  __shared__ half s_mean;
  __shared__ half s_variance;
  half2 mean = make_half2(__float2half(0.0f), __float2half(0.0f));
  half2 variance = make_half2(__float2half(0.0f), __float2half(0.0f));

  // calculate mean of the input.
  for (int i = tidx; i < n; i += blockDim.x) {
    const int idx = bidx * n + i;
    mean = __hadd2(mean, __ldg(&input[idx]));
  }
  mean = block_reduce_sum<half2>(mean);
  if (tidx == 0) {
    s_mean = __hdiv(__hadd(mean.x, mean.y), __float2half((float)n * 2));
  }
  __syncthreads();

  // calculate variance of the input.
  for (int i = tidx; i < n; i += blockDim.x) {
    const half2 x = __hsub2(input[bidx * n + i], make_half2(s_mean, s_mean));
    variance = __hadd2(variance, __hmul2(x, x));
  }
  variance = block_reduce_sum<half2>(variance);
  if (tidx == 0) {
    s_variance = __hadd(variance.x, variance.y);
    s_variance = __hdiv(s_variance, __float2half((float)n * 2));
    s_variance = __hadd(s_variance, __float2half(epsilon));
    s_variance = hrsqrt(s_variance);
  }
  __syncthreads();

  for (int i = tidx; i < n; i += blockDim.x) {
    const int idx = bidx * n + i;
    half2 local_out = __ldg(&input[idx]);
    local_out = __hsub2(local_out, make_half2(s_mean, s_mean));
    local_out = __hmul2(local_out, make_half2(s_variance, s_variance));
    local_out = __hmul2(local_out, __ldg(&weight[i]));
    if (bias != nullptr) {
      local_out = __hadd2(local_out, __ldg(&bias[i]));
    }
    out[idx] = local_out;
  }
}

void layer_norm(torch::Tensor& out,
                torch::Tensor input,
                torch::Tensor weight,
                torch::Tensor bias,
                float epsilon) {
  DCHECK(input.is_contiguous()) << "input tensor must be contiguous";
  DCHECK(out.is_contiguous()) << "output tensor must be contiguous";

  const int64_t n = input.size(1);

  dim3 grid(input.size(0));
  dim3 block(std::min<int>(n, 1024));
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

template <typename T>
void invoke_layernorm_kernel(T* out,
                             const T* input,
                             const T* weight,
                             const T* bias,
                             const float epsilon,
                             int m,
                             int n) {
  layer_norm_kernel<T><<<m, n>>>(out, input, weight, bias, epsilon, n);
}

template <>
void invoke_layernorm_kernel<half2>(half2* out,
                                    const half2* input,
                                    const half2* weight,
                                    const half2* bias,
                                    const float epsilon,
                                    int m,
                                    int n) {
  layer_norm_kernel<half2><<<m, n>>>(out, input, weight, bias, epsilon, n);
}
template <>
void invoke_layernorm_kernel<float>(float* out,
                                    const float* input,
                                    const float* weight,
                                    const float* bias,
                                    const float epsilon,
                                    int m,
                                    int n) {
  layer_norm_kernel<float><<<m, n>>>(out, input, weight, bias, epsilon, n);
}

}  // namespace llm::kernel