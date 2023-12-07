// adapted from https://github.com/NVIDIA/FasterTransformer
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include "../dispatch.h"

namespace llm::kernel {

template <typename T>
__global__ void apply_temperature_penalty_kernel(
    T* __restrict__ logits,
    const T* __restrict__ temperatures,
    int batch_size,
    int vocab_size) {
  // inverse temperatures for each batch
  extern __shared__ float inv_temperatures[];
  const int tid = threadIdx.x;
  // calculate inverse temperatures for each batch
  if (tid < batch_size) {
    // Replace 0. with 1. to avoid division by 0
    inv_temperatures[tid] =
        temperatures[tid] == 0 ? 1.0f : 1.0f / temperatures[tid];
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + tid; i < batch_size * vocab_size;
       i += blockDim.x * gridDim.x) {
    const int batch_idx = i / vocab_size;
    logits[i] *= inv_temperatures[batch_idx];
  }
}

void apply_temperature_penalty(torch::Tensor& logits,
                               const torch::Tensor& temperatures) {
  DCHECK(logits.is_contiguous()) << "logits tensor must be contiguous";
  DCHECK(temperatures.is_contiguous())
      << "temperatures tensor must be contiguous";

  const int batch_size = logits.size(0);
  const int vocab_size = logits.size(1);

  dim3 block(std::min(vocab_size, 1024));
  dim3 grid(std::min(batch_size * vocab_size / block.x, uint32_t(65536)));

  DISPATCH_FLOATING_TYPES(
      logits.scalar_type(), "apply_temperature_penalty_kernel", [&] {
        size_t smem_size = batch_size * sizeof(float);
        apply_temperature_penalty_kernel<scalar_t>
            <<<grid, block, smem_size, at::cuda::getCurrentCUDAStream()>>>(
                logits.data_ptr<scalar_t>(),
                temperatures.data_ptr<scalar_t>(),
                batch_size,
                vocab_size);
      });
}

template <typename T>
__global__ void apply_repetition_penalty_kernel(
    T* __restrict__ logits,
    const long* __restrict__ token_ids,
    const int* __restrict__ token_ids_lens,
    const T* __restrict__ penalities,
    int max_seq_len,
    int vocab_size) {
  const int tid = threadIdx.x;
  // batch idx
  const int bid = blockIdx.x;
  const float penalty = penalities[bid];
  const int len = token_ids_lens[bid];
  // move the pointer to the start of the batch
  logits += bid * vocab_size;

  for (int i = tid; i < len; i += blockDim.x) {
    const long token_id = token_ids[bid * max_seq_len + i];
    const float logit = logits[token_id];
    // assert(token_id < vocab_size);
    // apply repetition penalty
    logits[token_id] = logit < 0.0f ? logit * penalty : logit / penalty;
  }
}

void apply_repetition_penalty(torch::Tensor& logits,
                              const torch::Tensor& token_ids,
                              const torch::Tensor& token_ids_lens,
                              const torch::Tensor& penalities) {
  DCHECK(logits.is_contiguous()) << "logits tensor must be contiguous";
  DCHECK(token_ids.is_contiguous()) << "token_ids tensor must be contiguous";
  DCHECK(penalities.is_contiguous()) << "penalities tensor must be contiguous";
  DCHECK(logits.size(0) == token_ids.size(0))
      << "logits and token_ids must have the same batch size";

  const int batch_size = logits.size(0);
  const int vocab_size = logits.size(1);
  const int max_seq_len = token_ids.size(1);

  // each thread block handles one batch
  dim3 grid(batch_size);
  dim3 block(std::min(max_seq_len, 1024));

  DISPATCH_FLOATING_TYPES(
      logits.scalar_type(), "apply_repetition_penalty_kernel", [&] {
        apply_repetition_penalty_kernel<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                logits.data_ptr<scalar_t>(),
                token_ids.data_ptr<long>(),
                token_ids_lens.data_ptr<int>(),
                penalities.data_ptr<scalar_t>(),
                max_seq_len,
                vocab_size);
      });
}

template <typename T>
__global__ void apply_frequency_presence_penalty_kernel(
    T* __restrict__ logits,
    const long* __restrict__ token_ids,
    const int* __restrict__ token_counts,
    const int* __restrict__ token_ids_lens,
    const T* __restrict__ frequency_penalties,
    const T* __restrict__ presence_penalties,
    int max_seq_len,
    int vocab_size) {
  const int tid = threadIdx.x;
  // batch idx
  const int bid = blockIdx.x;
  const int len = token_ids_lens[bid];
  // move the pointer to the start of the batch
  logits += bid * vocab_size;

  for (int i = tid; i < len; i += blockDim.x) {
    const int idx = bid * max_seq_len + i;
    const long token_id = token_ids[idx];
    const int token_count = token_counts[idx];
    // assert(token_id < vocab_size);
    if (token_count > 0) {
      // apply frequency then presence penalities
      float logit = logits[token_id];
      logit -= (token_count * (float)frequency_penalties[bid]);
      logit -= presence_penalties[bid];
      logits[token_id] = logit;
    }
  }
}

void apply_frequency_presence_penalty(torch::Tensor& logits,
                                      const torch::Tensor& token_ids,
                                      const torch::Tensor& token_counts,
                                      const torch::Tensor& token_ids_lens,
                                      const torch::Tensor& frequency_penalties,
                                      const torch::Tensor& presence_penalties) {
  DCHECK(logits.is_contiguous()) << "logits tensor must be contiguous";
  DCHECK(token_ids.is_contiguous()) << "token_ids tensor must be contiguous";
  DCHECK(frequency_penalties.is_contiguous())
      << "penalities tensor must be contiguous";
  DCHECK(presence_penalties.is_contiguous())
      << "penalities tensor must be contiguous";
  DCHECK(logits.size(0) == token_ids.size(0))
      << "logits and token_ids must have the same batch size";

  const int batch_size = logits.size(0);
  const int vocab_size = logits.size(1);
  const int max_seq_len = token_ids.size(1);

  // each thread block handles one batch
  dim3 grid(batch_size);
  dim3 block(std::min(max_seq_len, 1024));

  DISPATCH_FLOATING_TYPES(
      logits.scalar_type(), "apply_frequency_presence_penalty_kernel", [&] {
        size_t smem_size = max_seq_len * (sizeof(float) + sizeof(int));
        apply_frequency_presence_penalty_kernel<scalar_t>
            <<<grid, block, smem_size, at::cuda::getCurrentCUDAStream()>>>(
                logits.data_ptr<scalar_t>(),
                token_ids.data_ptr<long>(),
                token_counts.data_ptr<int>(),
                token_ids_lens.data_ptr<int>(),
                frequency_penalties.data_ptr<scalar_t>(),
                presence_penalties.data_ptr<scalar_t>(),
                max_seq_len,
                vocab_size);
      });
}

}  // namespace llm::kernel
