// adapted from https://github.com/NVIDIA/FasterTransformer
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <cfloat>

#include "../dispatch.h"
#include "../reduce_kernel_utils.cuh"

namespace llm::kernel {

// Softmax(x_i) = exp(x_i - max_val) / sum(exp(x_j - max_val))
template <typename T>
__global__ void softmax_kernel(T* logits, int vocab_size) {
  const int tid = threadIdx.x;

  // move the pointer to the start of the batch
  logits += blockIdx.x * vocab_size;

  // use shared memory to save sum and max value
  __shared__ float s_sum_val;
  __shared__ float s_max_val;

  // get max value in the thread
  float max_val = -1 * FLT_MAX;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    max_val = max(max_val, logits[i]);
  }

  // get max value in the thread block and save it to shared memory
  max_val = block_reduce_max<float>(max_val);
  if (tid == 0) {
    s_max_val = max_val;
  }
  __syncthreads();

  float sum_val = 0.0f;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    logits[i] = __expf(logits[i] - s_max_val);
    sum_val += logits[i];
  }

  // get sum value in the thread block and save it to shared memory
  sum_val = block_reduce_sum<float>(sum_val);
  if (tid == 0) {
    // add a small epsilon to avoid division by zero
    s_sum_val = sum_val + 1e-6f;
  }
  __syncthreads();

  // compute softmax
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    logits[i] /= s_sum_val;
  }
}

// logits: [batch_size, vocab_size]
void invoke_softmax(torch::Tensor& logits) {
  DCHECK(logits.is_contiguous()) << "logits tensor must be contiguous";

  const int batch_size = logits.size(0);
  const int vocab_size = logits.size(1);

  // each thread block handles one batch
  dim3 grid(batch_size);
  dim3 block(std::min(vocab_size, 1024));

  DISPATCH_FLOATING_TYPES(logits.scalar_type(), "softmax_kernel", [&] {
    softmax_kernel<scalar_t>
        <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            logits.data_ptr<scalar_t>(), vocab_size);
  });
}

}  // namespace llm::kernel