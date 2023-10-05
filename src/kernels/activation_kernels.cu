// adapted from https://github.com/NVIDIA/FasterTransformer
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/TensorImpl.h>
#include <torch/extension.h>

#include "activation_kernels.h"

namespace llm::kernel {

/* Gelu Activation */

__forceinline__ __device__ float copysignf_pos(float a, float b) {
  float r;
  r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
  return r;
}

__inline__ __device__ float tanh_opt(float x) {
#if (__CUDA_ARCH__ >= 750 && CUDART_VERSION >= 11000)
  float r;
  asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  return r;
#else
  const float exp_val = -1.f * fabs(2 * x);
  return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
#endif
}

template <typename T>
struct GeluNewActivation {
  using return_type = T;
  static __device__ __forceinline__ T apply(const T& val) {
    const float cdf =
        0.5f * (1.0f + tanh_opt((0.7978845608028654f *
                                 (val + 0.044715f * val * val * val))));
    return val * cdf;
  }
};

template <typename T>
struct GeluFastActivation {
  using return_type = T;
  static __device__ __forceinline__ T apply(const T& val) {
    const float cdf = 0.5f * (1.0f + tanh_opt((0.7978845608028654f * val) *
                                              (1.0f + 0.044715f * val * val)));
    return val * cdf;
  }
};

/* Silu Activation */

template <typename T>
struct SiluActivation {
  using return_type = T;
  static __device__ __forceinline__ T apply(const T& val) {
    return (T)((float)val / (1.0f + __expf((float)-val)));
  }
};

template <template <typename T> class Activation, typename T>
__global__ void activation_kernel(T* output, const T* input, int stride) {
  for (int idx = blockIdx.x * stride + threadIdx.x; idx < stride;
       idx += blockDim.x) {
    output[idx] = Activation<T>::apply(input[idx]);
  }
}

// calculate act(x) * y where x = input[idx] and y = input[idx + stride]
template <template <typename T> class Activation, typename T>
__global__ void activation_and_mul_kernel(T* output,
                                          const T* input,
                                          int stride) {
  for (int idx = blockIdx.x * stride + threadIdx.x; idx < stride;
       idx += blockDim.x) {
    const T x = __ldg(&input[2 * idx]);
    const T y = __ldg(&input[2 * idx + stride]);
    output[idx] = Activation<T>::apply(x) * y;
  }
}

template <template <typename T> class Activation>
void launch_activation(torch::Tensor input, torch::Tensor& output) {
  const int stride = static_cast<int>(input.stride(0));
  dim3 grid(input.size(0));
  dim3 block(std::min(stride, 1024));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "activation_kernel",
      ([&] {
        activation_kernel<Activation, scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                stride);
      }));
}

template <template <typename T> class Activation>
void launch_activation_and_mul(torch::Tensor input, torch::Tensor& output) {
  const int stride = static_cast<int>(input.stride(0));
  dim3 grid(input.size(0));
  dim3 block(std::min(stride, 1024));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "activation_and_mul_kernel",
      ([&] {
        activation_and_mul_kernel<Activation, scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                stride);
      }));
}

torch::Tensor gelu_new(torch::Tensor input) {
  torch::Tensor output = torch::empty_like(input);
  launch_activation<GeluNewActivation>(input, output);
  return output;
}
torch::Tensor gelu_fast(torch::Tensor input) {
  torch::Tensor output = torch::empty_like(input);
  launch_activation<GeluFastActivation>(input, output);
  return output;
}
torch::Tensor silu(torch::Tensor input) {
  torch::Tensor output = torch::empty_like(input);
  launch_activation<SiluActivation>(input, output);
  return output;
}

}  // namespace llm::kernel
