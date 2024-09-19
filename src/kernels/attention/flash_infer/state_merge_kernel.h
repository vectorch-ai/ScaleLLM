//  Adapted from https://github.com/flashinfer-ai/flashinfer/
#pragma once

#include <cooperative_groups.h>

#include <flashinfer/attention/state.cuh>
#include <flashinfer/cp_async.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/utils.cuh>

namespace flashinfer {

template <uint32_t bdx, uint32_t bdy, uint32_t vec_size, typename DTypeIn>
__device__ __forceinline__ void threadblock_sync_state(state_t<vec_size>& st,
                                                       DTypeIn* v_smem,
                                                       float* s_smem) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t head_dim = vec_size * bdx;
  // [bdy, head_dim]
  st.o.cast_store(v_smem + ty * head_dim + tx * vec_size);
  // [bdy]
  s_smem[ty] = st.get_lse();
  st.init();
  __syncthreads();

#pragma unroll
  for (uint32_t iter = 0; iter < bdy; ++iter) {
    float s = s_smem[iter];
    vec_t<float, vec_size> v;
    v.cast_load(v_smem + iter * head_dim + tx * vec_size);
    st.merge(v, s, 1);
  }
}

template <uint32_t vec_size,
          uint32_t bdx,
          uint32_t bdy,
          uint32_t num_smem_stages,
          typename DTypeIn,
          typename DTypeOut,
          typename IdType>
__global__ void PersistentVariableLengthMergeStatesKernel(
    DTypeIn* __restrict__ V,
    float* __restrict__ S,
    IdType* indptr,
    DTypeOut* __restrict__ v_merged,
    float* __restrict__ s_merged,
    uint32_t seq_len,
    uint32_t num_heads) {
  using cp_async::PrefetchMode;
  using cp_async::SharedMemFillMode;

  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;
  uint32_t num_iters = ceil_div(seq_len * num_heads, num_ctas);
  constexpr uint32_t vec_bits = sizeof(DTypeIn) * vec_size * 8;
  constexpr uint32_t head_dim = vec_size * bdx;
  extern __shared__ uint8_t smem[];
  // [n_stages, bdy, head_dim]
  DTypeIn* v_smem = (DTypeIn*)smem;
  // [n_stages, bdy]
  float* s_smem =
      (float*)(smem + num_smem_stages * bdy * head_dim * sizeof(DTypeIn));

#pragma unroll 1
  for (uint32_t i = cta_id; i < seq_len * num_heads; i += num_ctas) {
    // token position
    uint32_t pos = i / num_heads;
    uint32_t head_idx = i % num_heads;
    state_t<vec_size> st;
    // is it possible that num_index_sets == 0?
    const uint32_t num_index_sets = indptr[pos + 1] - indptr[pos];

    if (num_index_sets == 0) {
      // fill with zeros
      vec_t<DTypeOut, vec_size> v;
      v.fill(DTypeOut(0.f));
      v.store(v_merged + (pos * num_heads + head_idx) * head_dim +
              tx * vec_size);
      if (s_merged != nullptr) {
        s_merged[pos * num_heads + head_idx] = -5e4;
      }
      continue;
    }

    if (num_index_sets == 1) {
      // copy over without merging
      vec_t<DTypeOut, vec_size> v;
      v.cast_load(V + (indptr[pos] * num_heads + head_idx) * head_dim +
                  tx * vec_size);
      v.store(v_merged + (pos * num_heads + head_idx) * head_dim +
              tx * vec_size);
      if (s_merged != nullptr) {
        s_merged[pos * num_heads + head_idx] =
            S[indptr[pos] * num_heads + head_idx];
      }
      continue;
    }

#pragma unroll
    for (uint32_t iter = 0; iter < num_smem_stages; ++iter) {
      cp_async::pred_load<vec_bits,
                          PrefetchMode::kPrefetch,
                          SharedMemFillMode::kNoFill>(
          v_smem + (iter * bdy + ty) * head_dim + tx * vec_size,
          V +
              ((indptr[pos] + (iter * bdy + ty)) * num_heads + head_idx) *
                  head_dim +
              tx * vec_size,
          (iter * bdy + ty) < num_index_sets);
      cp_async::commit_group();
    }
#pragma unroll 4
    for (uint32_t iter = 0; iter < ceil_div(num_index_sets, bdy); ++iter) {
      if (iter % bdx == 0) {
        s_smem[ty * bdx + tx] =
            iter * bdy + (ty * bdx + tx) < num_index_sets
                ? S[(indptr[pos] + (iter * bdy + ty * bdx + tx)) * num_heads +
                    head_idx]
                : 0.f;
        __syncthreads();
      }
      cp_async::wait_group<num_smem_stages - 1>();
      __syncthreads();

      vec_t<float, vec_size> v;
      v.cast_load(v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim +
                  tx * vec_size);
      if (iter * bdy + ty < num_index_sets) {
        float s = s_smem[(iter % bdx) * bdy + ty];
        st.merge(v, s, 1);
      }

      // wait for all threads to finish before prefetching the next stage
      __syncthreads();

      cp_async::pred_load<vec_bits,
                          PrefetchMode::kPrefetch,
                          SharedMemFillMode::kNoFill>(
          v_smem + ((iter % num_smem_stages) * bdy + ty) * head_dim +
              tx * vec_size,
          V +
              ((indptr[pos] + ((iter + num_smem_stages) * bdy + ty)) *
                   num_heads +
               head_idx) *
                  head_dim +
              tx * vec_size,
          (iter + num_smem_stages) * bdy + ty < num_index_sets);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    __syncthreads();

    st.normalize();
    // synchronize st within the threadblock by reusing the shared memory
    threadblock_sync_state<bdx, bdy, vec_size>(st, v_smem, s_smem);
    st.normalize();

    // write back the merged state and lse if needed
    // v_merged: [n_tokens, n_heads, head_dim]
    st.o.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim +
                    tx * vec_size);
    if (s_merged != nullptr) {
      // s_merged: [n_tokens, n_heads]
      s_merged[pos * num_heads + head_idx] = st.get_lse();
    }
  }
}

template <typename DTypeIn, typename DTypeOut, typename IdType>
cudaError_t VariableLengthMergeStates(DTypeIn* v,
                                      float* s,
                                      IdType* indptr,
                                      DTypeOut* v_merged,
                                      float* s_merged,
                                      uint32_t seq_len,
                                      uint32_t num_heads,
                                      uint32_t head_dim,
                                      cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size =
        std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    constexpr uint32_t bdx = HEAD_DIM / vec_size;
    constexpr uint32_t num_threads = 128;
    constexpr uint32_t bdy = num_threads / bdx;
    constexpr uint32_t num_smem_stages = 4;
    uint32_t smem_size = num_smem_stages * bdy * head_dim * sizeof(DTypeIn) +
                         num_threads * sizeof(float);
    auto kernel = PersistentVariableLengthMergeStatesKernel<vec_size,
                                                            bdx,
                                                            bdy,
                                                            num_smem_stages,
                                                            DTypeIn,
                                                            DTypeOut,
                                                            IdType>;
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm, kernel, num_threads, smem_size));
    num_blocks_per_sm =
        min(num_blocks_per_sm, ceil_div(seq_len * num_heads, num_sms));

    dim3 nblks(num_sms * num_blocks_per_sm);
    dim3 nthrs(bdx, bdy);
    void* args[] = {
        &v, &s, &indptr, &v_merged, &s_merged, &seq_len, &num_heads};
    FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(
        cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });
  return cudaSuccess;
}

}  // namespace flashinfer
