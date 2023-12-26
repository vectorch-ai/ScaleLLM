// adapted from https://github.com/NVIDIA/FasterTransformer

#include <curand_kernel.h>
#include <cub/cub.cuh>

namespace llm::kernel {

struct RunningTotalOp {
  // Running prefix
  float running_total;
  // Constructor
  __device__ RunningTotalOp(float running_total)
      : running_total(running_total) {}
  __device__ float operator()(float block_aggregate) {
    float old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};

template <typename T, int BLOCK_SIZE>
__global__ void topp_sampling(int* output_ids,
                              float* output_log_probs,
                              const int* __restrict sorted_ids,
                              const T* __restrict sorted_log_probs,
                              const float* __restrict top_ps,
                              int vocab_size,
                              curandState_t* curandstate) {
  // shared variables used to communicate between threads in a block
  // flag to indicate if the thread should stop scanning
  __shared__ int s_stop;
  // the random number generated by curand
  __shared__ float s_random_num;

  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = BLOCK_SIZE / kWarpSize;
  const int tid = threadIdx.x;
  const int batch_id = blockIdx.x;
  const int lane_id = tid % kWarpSize;
  const int warp_id = tid / kWarpSize;
  const float top_p = top_ps[batch_id];

  // let thread 0 to initialize the shared variables
  if (tid == 0) {
    s_stop = 0;
    s_random_num = curand_uniform(curandstate + batch_id) * top_p;
  }

  // TODO: quick path?

  // scan the sorted log probs to find the stopping position
  typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  // a shared variable to record which lane in each wrap has found the stopping
  // position
  __shared__ uint32_t s_selected_lane[kNumWarps];
  // a accumulative sum of the probs
  RunningTotalOp running_total_op(0.0f);

  // let lane 0 in each warp to initialize the shared variable
  if (lane_id == 0) {
    s_selected_lane[warp_id] = 0;
  }
  __syncthreads();

  const int offset = batch_id * vocab_size;
  const int end = (vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  float thread_log_prob = 0.0f;
  int active_idx = 0;
  for (int i = tid; i < end; i += BLOCK_SIZE) {
    const float log_prob =
        (i < vocab_size) ? static_cast<float>(sorted_log_probs[offset + i])
                         : 0.f;
    BlockScan(temp_storage)
        .InclusiveSum(log_prob, thread_log_prob, running_total_op);
    // gathers predicate bits from each thread in the warp
    const uint32_t lane_active_mask =
        __ballot_sync(0xFFFFFFFF, s_random_num <= thread_log_prob);

    active_idx = i;
    if (lane_active_mask != 0) {
      if (lane_id == 0) {
        atomicAdd(&s_stop, 1);
        s_selected_lane[warp_id] = lane_active_mask;
      }
    }
    __syncthreads();
    if (s_stop > 0) {
      break;
    }
  }

  // select first active warp
  bool skip = s_selected_lane[warp_id] == 0;
  for (int i = 1; i < warp_id; ++i) {
    if (s_selected_lane[i] != 0) {
      skip = true;
    }
  }

  if (!skip) {
    const int active_lane_id = kWarpSize - __popc(s_selected_lane[warp_id]);
    if (lane_id == active_lane_id) {
      output_ids[batch_id] = sorted_ids[offset + active_idx];
      const float log_prob = logf(sorted_log_probs[offset + active_idx]);
      output_log_probs[batch_id] = log_prob;
    }
  }
}

}  // namespace llm::kernel