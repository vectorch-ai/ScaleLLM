#pragma once
// This file contains utility functions for the CUDA kernels.
// ported from
// https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/reduce_kernel_utils.cuh

namespace llm::kernel {
// performs a parallel reduction operation across the threads within a single
// warp (32 threads).
//   - val: The value to be reduced within a warp.
template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
  // uses bitwise operations to perform a parallel reduction
  // within a warp. The 'mask' is right-shifted by 1 in each iteration
  // until it reaches zero, effectively summing all values within the warp.
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  }
  return val;
}

/* Calculate the sum of all elements in a thread block */
template <typename T>
__inline__ __device__ T block_reduce_sum(T val) {
  // up to 32 warps in a block
  static __shared__ T shared[32];
  // lane id in a warp
  int lane = threadIdx.x & 0x1f;
  // wrap id: threadIdx.x / 32
  int wid = threadIdx.x >> 5;

  // perform a parallel reduction across the threads within each warp
  val = warp_reduce_sum<T>(val);

  if (lane == 0) {
    // write the sum of each warp to shared memory
    shared[wid] = val;
  }
  // wait for all warps to finish
  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warp_reduce_sum<T>(val);
  return val;
}

}  // namespace llm::kernel
