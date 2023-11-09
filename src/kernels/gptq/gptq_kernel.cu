// adapted from
// https://github.com/PanQiWei/AutoGPTQ/blob/main/autogptq_extension/cuda_64/autogptq_cuda_kernel_64.cu

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/torch.h>

#include "common/logging.h"

namespace llm {

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) { return *reinterpret_cast<int*>(&i); }

template <typename T, int BLOCK_HEIGHT, int BLOCK_WIDTH, int BITS>
__global__ void vec_quant_matmul_kernel(const T* __restrict__ vec,
                                        const int* __restrict__ mat,
                                        T* __restrict__ mul,
                                        const T* __restrict__ scales,
                                        const int* __restrict__ zeros,
                                        const int* __restrict__ g_idx,
                                        int batch,
                                        int vec_height,
                                        int height,
                                        int width,
                                        int zero_width) {
  int h = BLOCK_HEIGHT * blockIdx.x;
  int w = BLOCK_WIDTH * blockIdx.y + threadIdx.x;

  __shared__ T blockvec[BLOCK_WIDTH];
  const int pack_factor = 32 / BITS;
  const int mask = (1 << BITS) - 1;
  int i = width * h + w;
  int g_h = h * pack_factor;
  int k;
  unsigned int g;
  T w_tmp;

  int z_w = w / pack_factor;
  int z_mod = (w % pack_factor) * BITS;

  float weight[BLOCK_WIDTH];

  for (k = 0; k < BLOCK_WIDTH; ++k) {
    int k_w = (k / pack_factor);
    int k_bit = (k % pack_factor) * BITS;

    g = as_int(g_idx[g_h + k]);
    T scale = scales[g * width + w];
    T zero = T((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & mask) + 1);

    w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & mask);

    weight[k] = scale * (w_tmp - zero);
  }

  T res;
  for (int b = 0; b < batch; ++b) {
    res = 0;

    blockvec[threadIdx.x] =
        vec[b * vec_height + blockIdx.x * BLOCK_WIDTH + threadIdx.x];
    __syncthreads();
    for (k = 0; k < BLOCK_WIDTH; ++k) {
      res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
}

template <int BLOCK_HEIGHT, int BLOCK_WIDTH, int BITS>
void vec_quant_matmul_launch_kernel(torch::Tensor vec,
                                    torch::Tensor mat,
                                    torch::Tensor mul,
                                    torch::Tensor scales,
                                    torch::Tensor zeros,
                                    torch::Tensor g_idx) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks((height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
              (width + BLOCK_WIDTH - 1) / BLOCK_WIDTH);
  dim3 threads(BLOCK_WIDTH);

  AT_DISPATCH_FLOATING_TYPES(
      vec.scalar_type(), "vec_quant_matmul_kernel", ([&] {
        vec_quant_matmul_kernel<scalar_t, BLOCK_HEIGHT, BLOCK_WIDTH, BITS>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                vec.data_ptr<scalar_t>(),
                mat.data_ptr<int>(),
                mul.data_ptr<scalar_t>(),
                scales.data_ptr<scalar_t>(),
                zeros.data_ptr<int>(),
                g_idx.data_ptr<int>(),
                batch,
                vec_height,
                height,
                width,
                zero_width);
      }));
}

template <typename T, int BLOCK_HEIGHT, int BLOCK_WIDTH>
__global__ void vec_quant3_matmul_kernel(const T* __restrict__ vec,
                                         const int* __restrict__ mat,
                                         T* __restrict__ mul,
                                         const T* __restrict__ scales,
                                         const int* __restrict__ zeros,
                                         const int* __restrict__ g_idx,
                                         int batch,
                                         int vec_height,
                                         int height,
                                         int width,
                                         int zero_width) {
  int h = BLOCK_HEIGHT * blockIdx.x;
  int w = BLOCK_WIDTH * blockIdx.y + threadIdx.x;

  __shared__ T blockvec[BLOCK_WIDTH];
  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k;
  unsigned int g;
  T w_tmp;

  int z_w = (w / 32) * 3;
  int z_mod = w % 32;
  int z_bit;
  unsigned int z_tmp;
  if (z_mod != 10) {
    if (z_mod != 21) {
      z_bit = z_mod;
      if (z_bit > 21) {
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10) {
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }

  float weight[BLOCK_WIDTH];

  for (k = 0; k < BLOCK_WIDTH; ++k) {
    int k_w = (k / 32) * 3;
    int k_mod = k % 32;
    int k_bit;

    if (k_mod != 10) {
      if (k_mod != 21) {
        k_bit = k_mod;
        if (k_bit > 21) {
          k_bit -= 22;
          k_bit *= 3;
          k_bit += 2;
          k_w += 2;
        } else if (k_bit > 10) {
          k_bit -= 11;
          k_bit *= 3;
          k_bit += 1;
          k_w += 1;
        } else {
          k_bit *= 3;
        }
      } else {
        k_w += 1;
      }
    }

    g = as_int(g_idx[g_h + k]);
    T scale = scales[g * width + w];
    T zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) |
              ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = T((z_tmp) + 1);
    } else if (z_mod == 21) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) |
              ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = T((z_tmp) + 1);
    } else {
      zero = T(((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1);
    }

    if (k_mod == 10) {
      w_tmp = (as_unsigned(mat[i + (k_w * width)]) >> 30) |
              ((as_unsigned(mat[i + ((k_w + 1) * width)]) << 2) & 0x4);
    } else if (k_mod == 21) {
      w_tmp = (as_unsigned(mat[i + (k_w * width)]) >> 31) |
              ((as_unsigned(mat[i + ((k_w + 1) * width)]) << 1) & 0x6);
    } else {
      w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0x7);
    }
    weight[k] = scale * (w_tmp - zero);
  }

  T res;
  for (int b = 0; b < batch; ++b) {
    res = 0;

    blockvec[threadIdx.x] =
        vec[b * vec_height + blockIdx.x * BLOCK_WIDTH + threadIdx.x];
    __syncthreads();
    for (k = 0; k < BLOCK_WIDTH; ++k) {
      res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
}

template <int BLOCK_HEIGHT, int BLOCK_WIDTH>
void vec_quant3_matmul_launch_kernel(torch::Tensor vec,
                                     torch::Tensor mat,
                                     torch::Tensor mul,
                                     torch::Tensor scales,
                                     torch::Tensor zeros,
                                     torch::Tensor g_idx) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks((height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT,
              (width + BLOCK_WIDTH - 1) / BLOCK_WIDTH);
  dim3 threads(BLOCK_WIDTH);

  AT_DISPATCH_FLOATING_TYPES(
      vec.scalar_type(), "vec_quant3_matmul_kernel", ([&] {
        vec_quant3_matmul_kernel<scalar_t, BLOCK_HEIGHT, BLOCK_WIDTH>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                vec.data_ptr<scalar_t>(),
                mat.data_ptr<int>(),
                mul.data_ptr<scalar_t>(),
                scales.data_ptr<scalar_t>(),
                zeros.data_ptr<int>(),
                g_idx.data_ptr<int>(),
                batch,
                vec_height,
                height,
                width,
                zero_width);
      }));
}

void vec_quant_matmul_64(torch::Tensor vec,
                         torch::Tensor mat,
                         torch::Tensor mul,
                         torch::Tensor scales,
                         torch::Tensor zeros,
                         torch::Tensor g_idx,
                         int64_t bits) {
  switch (bits) {
    case 2:
      return vec_quant_matmul_launch_kernel</*BLOCK_HEIGHT=*/4,
                                            /*BLOCK_WIDTH=*/64,
                                            /*BITS=*/2>(
          vec, mat, mul, scales, zeros, g_idx);
    case 3:
      return vec_quant3_matmul_launch_kernel</*BLOCK_HEIGHT=*/6,
                                             /*BLOCK_WIDTH=*/64>(
          vec, mat, mul, scales, zeros, g_idx);
    case 4:
      return vec_quant_matmul_launch_kernel</*BLOCK_HEIGHT=*/8,
                                            /*BLOCK_WIDTH=*/64,
                                            /*BITS=*/4>(
          vec, mat, mul, scales, zeros, g_idx);
    case 8:
      return vec_quant_matmul_launch_kernel</*BLOCK_HEIGHT=*/16,
                                            /*BLOCK_WIDTH=*/64,
                                            /*BITS=*/8>(
          vec, mat, mul, scales, zeros, g_idx);
    default:
      GLOG(FATAL) << "Unsupported bits " << bits;
  }
  __builtin_unreachable();
}

void vec_quant_matmul_256(torch::Tensor vec,
                          torch::Tensor mat,
                          torch::Tensor mul,
                          torch::Tensor scales,
                          torch::Tensor zeros,
                          torch::Tensor g_idx,
                          int64_t bits) {
  switch (bits) {
    case 2:
      return vec_quant_matmul_launch_kernel</*BLOCK_HEIGHT=*/16,
                                            /*BLOCK_WIDTH=*/256,
                                            /*BITS=*/2>(
          vec, mat, mul, scales, zeros, g_idx);
    case 3:
      return vec_quant3_matmul_launch_kernel</*BLOCK_HEIGHT=*/24,
                                             /*BLOCK_WIDTH=*/256>(
          vec, mat, mul, scales, zeros, g_idx);
    case 4:
      return vec_quant_matmul_launch_kernel</*BLOCK_HEIGHT=*/32,
                                            /*BLOCK_WIDTH=*/256,
                                            /*BITS=*/4>(
          vec, mat, mul, scales, zeros, g_idx);
    case 8:
      return vec_quant_matmul_launch_kernel</*BLOCK_HEIGHT=*/64,
                                            /*BLOCK_WIDTH=*/256,
                                            /*BITS=*/8>(
          vec, mat, mul, scales, zeros, g_idx);
    default:
      GLOG(FATAL) << "Unsupported bits " << bits;
  }
  __builtin_unreachable();
}

}  // namespace llm
