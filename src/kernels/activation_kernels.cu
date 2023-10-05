// adapted from https://github.com/NVIDIA/FasterTransformer

#include "activation_kernels.h"

namespace llm {

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
    const float cdf =
        0.5f * (1.0f + tanh_opt((0.7978845608028654f * val) *
                                 (1.0f + 0.044715f * val * val)));
    return val * cdf;
  }
};

/* Relu Activation */
template <typename T>
struct ReluActivation {
  using return_type = T;
  static __device__ __forceinline__ T apply(const T& val) {
    return val > static_cast<T>(0.0f) ? val : static_cast<T>(0.0f);
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

}  // namespace llm
