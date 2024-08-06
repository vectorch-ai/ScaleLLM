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

template <typename T>
__global__ void gemma_rms_norm_kernel(T* __restrict__ out,
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

  // Llama does x.to(float16) * w whilst Gemma2 is (x * (w + 1)).to(float16)
  // See https://github.com/huggingface/transformers/pull/29402
  for (int64_t i = tidx; i < n; i += blockDim.x) {
    const int64_t idx = bidx * n + i;
    const float x = input[idx];
    const float w = weight[i];
    out[idx] = (T)(x * s_variance * (1.0 + w));
  }
}

void gemma_rms_norm(torch::Tensor& out,
                    torch::Tensor input,
                    torch::Tensor weight,
                    float epsilon) {
  DCHECK(input.is_contiguous()) << "input tensor must be contiguous";
  DCHECK(out.is_contiguous()) << "output tensor must be contiguous";

  const int64_t n = input.size(1);

  dim3 grid(input.size(0));
  dim3 block(std::min<int>(n, 1024));
  DISPATCH_FLOATING_TYPES(input.scalar_type(), "rms_norm_kernel", [&] {
    gemma_rms_norm_kernel<scalar_t>
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

}  // namespace llm::kernel
