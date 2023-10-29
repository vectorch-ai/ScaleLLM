#include <ATen/cuda/CUDAContext.h>
#include <c10/core/TensorImpl.h>
#include <torch/torch.h>

#include "activation_kernels.h"
#include "common/logging.h"
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
                                  int n,
                                  int stride) {
  const uint32_t src_base_idx = blockIdx.x * stride;
  const uint32_t dst_base_idx = blockIdx.x * n;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const T x = __ldg(&input[src_base_idx + i]);
    out[dst_base_idx + i] = Activation<T>::apply(x);
  }
}

template <template <typename T> class Activation>
void launch_activation(torch::Tensor& out, torch::Tensor input) {
  const int n = static_cast<int>(input.size(1));
  const int stride = static_cast<int>(input.stride(0));
  dim3 grid(input.size(0));
  dim3 block(std::min(n, 1024));
  DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "activation_kernel", ([&] {
        activation_kernel<Activation, scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                out.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                n,
                stride);
      }));
}

// calculate act(x) * y where x = input[idx] and y = input[idx + n]
template <template <typename T> class Activation, typename T>
__global__ void activation_and_mul_kernel(T* __restrict__ out,
                                          const T* __restrict__ input,
                                          int n) {
  const uint32_t base_idx = blockIdx.x * n;
  for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
    const uint32_t i_idx = 2 * base_idx + i;
    const T x = __ldg(&input[i_idx]);
    const T y = __ldg(&input[i_idx + n]);
    out[base_idx + i] = Activation<T>::apply(x) * y;
  }
}

template <template <typename T> class Activation>
void launch_activation_and_mul(torch::Tensor& out, torch::Tensor input) {
  const int n = static_cast<int>(input.size(1)) / 2;
  dim3 grid(input.size(0));
  dim3 block(std::min(n, 1024));
  DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "activation_and_mul_kernel", ([&] {
        activation_and_mul_kernel<Activation, scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), n);
      }));
}

}  // namespace

torch::Tensor gelu_new(torch::Tensor input) {
  torch::Tensor out = torch::empty_like(input);
  launch_activation<GeluNewActivation>(out, input);
  return out;
}
torch::Tensor gelu_fast(torch::Tensor input) {
  torch::Tensor out = torch::empty_like(input);
  launch_activation<GeluFastActivation>(out, input);
  return out;
}
torch::Tensor silu(torch::Tensor input) {
  torch::Tensor out = torch::empty_like(input);
  launch_activation<SiluActivation>(out, input);
  return out;
}

// calculate act(x) * y where x = input[0] and y = input[1]
torch::Tensor gelu_new_with_mul(torch::Tensor input) {
  GCHECK(input.is_contiguous());
  std::vector<int64_t> sizes = input.sizes().vec();
  GCHECK(sizes.size() == 2);
  sizes[1] /= 2;
  torch::Tensor out = torch::empty(sizes, input.options());
  launch_activation_and_mul<GeluNewActivation>(out, input);
  return out;
}
torch::Tensor gelu_fast_with_mul(torch::Tensor input) {
  GCHECK(input.is_contiguous());
  std::vector<int64_t> sizes = input.sizes().vec();
  GCHECK(sizes.size() == 2);
  sizes[1] /= 2;
  torch::Tensor out = torch::empty(sizes, input.options());
  launch_activation_and_mul<GeluFastActivation>(out, input);
  return out;
}
torch::Tensor silu_with_mul(torch::Tensor input) {
  GCHECK(input.is_contiguous());
  std::vector<int64_t> sizes = input.sizes().vec();
  GCHECK(sizes.size() == 2);
  sizes[1] /= 2;
  torch::Tensor out = torch::empty(sizes, input.options());
  launch_activation_and_mul<SiluActivation>(out, input);
  return out;
}

}  // namespace llm::kernel
