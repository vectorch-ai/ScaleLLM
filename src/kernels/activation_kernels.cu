#include <ATen/cuda/CUDAContext.h>
#include <c10/core/TensorImpl.h>
#include <torch/extension.h>

#include "activation_kernels.h"
#include "dispatch.h"

namespace llm::kernel {
namespace {

/* Gelu Activation */
// adapted from https://github.com/NVIDIA/FasterTransformer
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
    // x * sigmoid(x)
    return (T)((float)val / (1.0f + __expf((float)-val)));
  }
};

template <template <typename T> class Activation, typename T>
__global__ void activation_kernel(T* __restrict__ out,
                                  const T* __restrict__ input,
                                  int stride) {
  const uint32_t base_idx = blockIdx.x * stride;
  for (uint32_t i = threadIdx.x; i < stride; i += blockDim.x) {
    const uint32_t idx = base_idx + i;
    const T x = __ldg(&input[idx]);
    out[idx] = Activation<T>::apply(x);
  }
}

// calculate act(x) * y where x = input[idx] and y = input[idx + stride]
template <template <typename T> class Activation, typename T>
__global__ void activation_and_mul_kernel(T* __restrict__ out,
                                          const T* __restrict__ input,
                                          int stride) {
  const uint32_t base_idx = blockIdx.x * stride;
  for (uint32_t i = threadIdx.x; i < stride; i += blockDim.x) {
    const uint32_t i_idx = 2 * base_idx + i;
    const T x = __ldg(&input[i_idx]);
    const T y = __ldg(&input[i_idx + stride]);
    out[base_idx + i] = Activation<T>::apply(x) * y;
  }
}

template <template <typename T> class Activation>
void launch_activation(torch::Tensor& out, torch::Tensor input) {
  const int stride = static_cast<int>(input.stride(0));
  dim3 grid(input.size(0));
  dim3 block(std::min(stride, 1024));
  DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "activation_kernel", ([&] {
        activation_kernel<Activation, scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                stride);
      }));
}

template <template <typename T> class Activation>
void launch_activation_and_mul(torch::Tensor& out, torch::Tensor input) {
  const int stride = static_cast<int>(input.stride(0));
  dim3 grid(input.size(0));
  dim3 block(std::min(stride, 1024));
  DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "activation_and_mul_kernel", ([&] {
        activation_and_mul_kernel<Activation, scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                stride);
      }));
}

}  // namespace

torch::Tensor gelu_new(torch::Tensor input) {
  CHECK(input.is_contiguous()) << "input tensor must be contiguous";

  torch::Tensor out = torch::empty_like(input);
  launch_activation<GeluNewActivation>(out, input);
  return out;
}
torch::Tensor gelu_fast(torch::Tensor input) {
  CHECK(input.is_contiguous()) << "input tensor must be contiguous";

  torch::Tensor out = torch::empty_like(input);
  launch_activation<GeluFastActivation>(out, input);
  return out;
}
torch::Tensor silu(torch::Tensor input) {
  CHECK(input.is_contiguous()) << "input tensor must be contiguous";

  torch::Tensor out = torch::empty_like(input);
  launch_activation<SiluActivation>(out, input);
  return out;
}

}  // namespace llm::kernel
