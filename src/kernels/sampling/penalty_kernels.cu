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
    // add a small epsilon to avoid division by zero
    inv_temperatures[tid] = 1.0f / (temperatures[tid] + 1e-6f);
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + tid; i < batch_size * vocab_size;
       i += blockDim.x * gridDim.x) {
    const int batch_idx = i / vocab_size;
    logits[i] *= inv_temperatures[batch_idx];
  }
}

void apply_temperature_penalty(torch::Tensor& logits,
                               torch::Tensor temperatures) {
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

}  // namespace llm::kernel
