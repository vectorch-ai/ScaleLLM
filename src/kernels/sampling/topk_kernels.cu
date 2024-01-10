// adapted from https://github.com/NVIDIA/FasterTransformer

#include <ATen/cuda/CUDAContext.h>
#include <curand_kernel.h>
#include <torch/torch.h>

#include <cub/cub.cuh>

#include "../dispatch.h"
#include "../reduce_kernel_utils.cuh"
#include "common/logging.h"

namespace llm::kernel {

// reduce topk for each thread block and save the result to temp storage
template <typename T, int BLOCK_SIZE, int BLOCKS_PER_SEQ>
__global__ void partial_topk_within_block(const T* __restrict logits,
                                          T* __restrict tmp_logits,
                                          int* __restrict tmp_topk_ids,
                                          T* __restrict tmp_topk_logits,
                                          int max_top_k,
                                          const int* __restrict top_ks,
                                          int vocab_size) {
  typedef cub::BlockReduce<TopK_2<T>, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  // one sequence is splited into BLOCKS_PER_SEQ blocks for parallel processing
  const int batch_id = bid / BLOCKS_PER_SEQ;
  // block lane for the sequence
  const int block_lane = bid % BLOCKS_PER_SEQ;
  const int k = top_ks[batch_id];

  const int tmp_logits_base = batch_id * vocab_size;
  const int tmp_topk_base =
      batch_id * BLOCKS_PER_SEQ * max_top_k + block_lane * k;

  // copy log_probs to tmp_log_probs for modifying
#pragma unroll
  for (int id = tid + block_lane * BLOCK_SIZE; id < vocab_size;
       id += BLOCK_SIZE * BLOCKS_PER_SEQ) {
    const int index = id + tmp_logits_base;
    tmp_logits[index] = logits[index];
  }

  TopK_2<T> partial;
  const T MAX_T_VAL = (std::is_same_v<T, half>) ? HALF_FLT_MAX : FLT_MAX;

  // every thread does top-k then combine the result with block reduce
  for (int ite = 0; ite < k; ite++) {
    partial.init();

#pragma unroll
    for (int id = tid + block_lane * BLOCK_SIZE; id < vocab_size;
         id += BLOCK_SIZE * BLOCKS_PER_SEQ) {
      const int index = id + tmp_logits_base;
      partial.insert(tmp_logits[index], index);
    }

    // reduce within each block
    TopK_2<T> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<T>);

    // save the topk idx and value to temp storage
    if (tid == 0) {
      const int index = tmp_topk_base + ite;
      tmp_topk_ids[index] = total.p;
      tmp_topk_logits[index] = total.u;
      // remove the largest item by setting the score to -MAX_T_VAL
      tmp_logits[total.p] = -MAX_T_VAL;
    }

    // wait for all threads to finish
    __syncthreads();
  }
}

// reduce topk across blocks for each batch
template <typename T, int BLOCK_SIZE, int BLOCKS_PER_SEQ>
__global__ void topk_sampling_across_blocks(int* output_ids,
                                            float* output_log_probs,
                                            const int* __restrict tmp_topk_ids,
                                            T* __restrict tmp_topk_logits,
                                            int max_top_k,
                                            const int* __restrict top_ks,
                                            const float* __restrict top_ps,
                                            curandState_t* curandstate) {
  typedef cub::BlockReduce<TopK_2<float>, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const T MAX_T_VAL = (std::is_same_v<T, half>) ? HALF_FLT_MAX : FLT_MAX;

  const int tid = threadIdx.x;
  // each block processes one sequence
  const int batch_id = blockIdx.x;
  const int k = top_ks[batch_id];
  const float p = top_ps != nullptr ? top_ps[batch_id] : 1.0f;
  const int stride = max_top_k * BLOCKS_PER_SEQ;

  // move the pointer to the corresponding batch
  T* topk_logits = tmp_topk_logits + batch_id * stride;
  const int* topk_ids = tmp_topk_ids + batch_id * stride;

  // use shared memory to save temp topk idxs and values
  extern __shared__ char smem[];  // idxs + vals for topk
  int* s_idxs = reinterpret_cast<int*>(smem);
  float* s_vals = reinterpret_cast<float*>(s_idxs + k);

  // use shared memory to save sum and max value for softmax
  __shared__ float s_sum_val;
  __shared__ float s_max_val;
  if (tid == 0) {
    // add a small epsilon to avoid division by zero
    s_sum_val = 1e-6f;
  }

  // each block has a partial topk
  const int size = k * BLOCKS_PER_SEQ;
  // use float to record laggest value
  TopK_2<float> partial;
  // calculate topk and softmax for each sequence
  for (int ite = 0; ite < k; ++ite) {
    partial.init();

    // merge partial topk from all blocks
#pragma unroll
    for (int i = tid; i < size; i += BLOCK_SIZE) {
      partial.insert(topk_logits[i], i);
    }

    // reduce within each block to get the top idx and value
    TopK_2<float> total =
        BlockReduce(temp_storage).Reduce(partial, reduce_topk_op_2<float>);

    // save the topk idx and value to shared memory
    if (tid == 0) {
      if (ite == 0) {
        s_max_val = total.u;
      }
      s_idxs[ite] = total.p;
      // remove the largest item by setting the score to -MAX_T_VAL
      topk_logits[total.p] = -MAX_T_VAL;

      // calculate expf(x - max_val) and sum for softmax
      const float exp_logit = __expf(total.u - s_max_val);
      s_vals[ite] = exp_logit;
      s_sum_val += exp_logit;
    }
    __syncthreads();
  }

  // let thread 0 sample the id from topk candidates
  if (tid == 0) {
    float rand_num = curand_uniform(curandstate + batch_id) * p * s_sum_val;
    for (int i = 0; i < k; ++i) {
      const float exp_logit = s_vals[i];
      rand_num -= exp_logit;
      if (rand_num <= 0 || i == k - 1) {
        output_ids[batch_id] = topk_ids[s_idxs[i]];
        // the log_prob is the probability of the selected tokens
        const float log_prob = logf(exp_logit / s_sum_val);
        output_log_probs[batch_id] = log_prob;
        break;
      }
    }
  }
}

// topk sampling kernel launcher that calculates the topk for each sequence in
// following steps:
// 1. split the sequence into BLOCKS_PER_SEQ blocks for parallel processing and
// calculate the partial topk for each block
// 2. reduce the partial topk across blocks for each sequence
#define CASE_K_RANGE(K_MIN, K_MAX, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCKS_PER_SEQ) \
  case K_MIN ... K_MAX:                                                        \
    partial_topk_within_block<scalar_t, BLOCK_SIZE_1, BLOCKS_PER_SEQ>          \
        <<<batch_size * BLOCKS_PER_SEQ, BLOCK_SIZE_1, 0, stream>>>(            \
            _logits,                                                           \
            tmp_logits,                                                        \
            tmp_topk_ids,                                                      \
            tmp_topk_logits,                                                   \
            max_top_k,                                                         \
            _top_ks,                                                           \
            vocab_size);                                                       \
    topk_sampling_across_blocks<scalar_t, BLOCK_SIZE_2, BLOCKS_PER_SEQ>        \
        <<<batch_size,                                                         \
           BLOCK_SIZE_2,                                                       \
           K_MAX * sizeof(int) + K_MAX * sizeof(float),                        \
           stream>>>(_output_ids,                                              \
                     _output_log_probs,                                        \
                     tmp_topk_ids,                                             \
                     tmp_topk_logits,                                          \
                     max_top_k,                                                \
                     _top_ks,                                                  \
                     _top_ps,                                                  \
                     curandstate);                                             \
    break;

void invoke_topk_sampling(torch::Tensor& output_ids,
                          torch::Tensor& output_log_probs,
                          torch::Tensor logits,
                          torch::Tensor workspace,
                          curandState_t* curandstate,
                          int max_top_k,
                          torch::Tensor top_ks,
                          torch::Tensor top_ps) {
  const int batch_size = logits.size(0);
  const int vocab_size = logits.size(1);
  const int max_blocks_per_seq = 8;

  // tmp_logits to save modified logits
  size_t tmp_logits_size = batch_size * vocab_size;
  // tmp_topk_* to save topk ids and logits for each block
  size_t tmp_topk_size = batch_size * max_blocks_per_seq * max_top_k;
  // round up to prevent memory misalignment
  tmp_logits_size = ((tmp_logits_size + 3) / 4) * 4;
  tmp_topk_size = ((tmp_topk_size + 3) / 4) * 4;

  DISPATCH_FLOATING_TYPES(logits.scalar_type(), "tok_kernel", [&] {
    const size_t min_workspace_size =
        tmp_logits_size * sizeof(scalar_t) +
        tmp_topk_size * (sizeof(int) + sizeof(scalar_t));
    assert(workspace.numel() >= min_workspace_size);

    // scratch space for topk
    scalar_t* tmp_logits = workspace.data_ptr<scalar_t>();
    int* tmp_topk_ids = reinterpret_cast<int*>(tmp_logits + tmp_logits_size);
    scalar_t* tmp_topk_logits =
        reinterpret_cast<scalar_t*>(tmp_topk_ids + tmp_topk_size);

    int* _output_ids = output_ids.data_ptr<int>();
    float* _output_log_probs = output_log_probs.data_ptr<float>();

    const scalar_t* _logits = logits.data_ptr<scalar_t>();
    const int* _top_ks = top_ks.data_ptr<int>();
    const float* _top_ps =
        top_ps.defined() ? top_ps.data_ptr<float>() : nullptr;

    auto stream = at::cuda::getCurrentCUDAStream();
    switch (max_top_k) {
      // K_MIN, K_MAX, BLOCK_SIZE_1, BLOCK_SIZE_2, BLOCKS_PER_SEQ
      CASE_K_RANGE(1, 16, 128, 128, 8);
      CASE_K_RANGE(17, 32, 256, 128, 8);
      CASE_K_RANGE(33, 64, 256, 256, 8);
      CASE_K_RANGE(65, 1024, 256, 256, 8);
      default:
        GLOG(FATAL) << "topk_sampling only supports max_top_k <= 1024 but got "
                    << max_top_k;
    }
  });
}
#undef CASE_K

}  // namespace llm::kernel