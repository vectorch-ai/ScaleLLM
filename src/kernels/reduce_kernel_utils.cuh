#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// This file contains utility functions for the CUDA kernels.
// ported from https://github.com/NVIDIA/FasterTransformer

namespace llm::kernel {
constexpr unsigned FINAL_MASK = 0xffffffff;
constexpr float HALF_FLT_MAX = 65504.0f;

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
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}

// performs a parallel reduction operation across the threads within a single
// warp (32 threads).
//   - val: The value to be reduced within a warp.
template <typename T>
__inline__ __device__ T warp_reduce_max(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
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

/* Calculate the max of all elements in a thread block */
template <typename T>
__inline__ __device__ T block_reduce_max(T val) {
  // up to 32 warps in a block
  static __shared__ T shared[32];
  // lane id in a warp
  int lane = threadIdx.x & 0x1f;
  // wrap id: threadIdx.x / 32
  int wid = threadIdx.x >> 5;

  // get max value in each warp
  val = warp_reduce_max<T>(val);

  // record in-warp max value to shared memory with warp id
  if (lane == 0) {
    // write the sum of each warp to shared memory
    shared[wid] = val;
  }
  // wait for all warps to finish
  __syncthreads();

  // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
  // blockDim.x is not divided by 32
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
  val = warp_reduce_max<T>(val);
  return val;
}

// data struct for topk kernels
// An heap is used to store the topk elements. the heap is using bubble sort to
// maintain the order.
template <typename T, int K>
struct TopK {
  // the index of the topk elements
  int p[K];
  // the value of the topk elements
  T u[K];

  // insert an element into the heap and maintain the order via bubble sort
  __device__ __forceinline__ void insert(T elem, int elem_id) {
    // replace the last element with the new element if the new element is
    // larger than the last element.
    if (elem > u[K - 1] || (p[K - 1] == -1) ||
        ((elem == u[K - 1]) && (elem_id < p[K - 1]))) {
      u[K - 1] = elem;
      p[K - 1] = elem_id;
    }

    // bubble sort to maintain the order
    for (int k = K - 2; k >= 0; --k) {
      if ((u[k + 1] > u[k]) || (p[k] == -1) ||
          ((u[k + 1] == u[k]) && (p[k + 1] < p[k]))) {
        T u2 = u[k];
        int p2 = p[k];
        u[k] = u[k + 1];
        p[k] = p[k + 1];
        u[k + 1] = u2;
        p[k + 1] = p2;
      }
    }
  }

  // initialize the heap with pointers to -1 and values to -FLT_MAX
  __device__ __forceinline__ void init() {
    const bool IS_FP16 = std::is_same_v<T, half>;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

    for (int i = 0; i < K; i++) {
      p[i] = -1;
      u[i] = -MAX_T_VAL;
    }
  }
};

// operator for cub::BlockReduce to get topk across a thread block
template <typename T, int K>
__device__ __forceinline__ TopK<T, K> reduce_topk_op(
    const TopK<T, K>& a,
    const TopK<T, K>& b) {
  TopK<T, K> res = a;
  for (int i = 0; i < K; ++i) {
    res.insert(b.u[i], b.p[i]);
  }
  return res;
}

// Similar to TopK, but only store the largest element
template <typename T>
struct TopK_2 {
  // the index of the topk elements
  int p = -1;
  // the value of the topk elements
  T u = -((std::is_same_v<T, half>) ? HALF_FLT_MAX : FLT_MAX);

  __device__ __forceinline__ void insert(T elem, int elem_id) {
    if (elem > u) {
      u = elem;
      p = elem_id;
    }
  }

  __device__ __forceinline__ void init() {
    p = -1;
    u = -((std::is_same_v<T, half>) ? HALF_FLT_MAX : FLT_MAX);
  }
};

// operator for cub::BlockReduce to get largest element across a thread block
template <typename T>
__device__ __forceinline__ TopK_2<T> reduce_topk_op_2(const TopK_2<T>& a,
                                                      const TopK_2<T>& b) {
  return a.u > b.u ? a : b;
}

}  // namespace llm::kernel
