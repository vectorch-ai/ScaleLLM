#include "model_runner.h"

#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "memory/kv_cache.h"
#include "models/causal_lm.h"
#include "models/parameters.h"

namespace llm {

const static std::vector<int64_t> kBatchSizesForCudaGraph = {
    1,   2,   4,   8,   16,  24,  32,  40,  48,  56,  64,  72,
    80,  88,  96,  104, 112, 120, 128, 136, 144, 152, 160, 168,
    176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256};

// TODO: pass in as parameters
constexpr int64_t k_max_seq_len = 1024;
constexpr int64_t block_size = 16;

// capture graph with batch size list
void ModelRunner::capture_graphs(std::vector<KVCache>& kv_cache) {
  std::vector<int64_t> batch_sizes = kBatchSizesForCudaGraph;
  CHECK(!batch_sizes.empty()) << "batch_sizes is empty";

  // sort batch_sizes in descending order
  std::sort(batch_sizes.begin(), batch_sizes.end(), std::greater<int64_t>());

  torch::DeviceGuard device_guard(device_);
  // allocate tensors for sharing among graphs
  auto options = torch::dtype(torch::kInt32).device(device_);
  const int64_t max_batch_size = batch_sizes.front();
  torch::Tensor token_ids = torch::zeros({max_batch_size}, options);
  torch::Tensor positions = torch::zeros({max_batch_size}, options);
  torch::Tensor q_cu_seq_lens = torch::range(
      /*start=*/0, /*end=*/max_batch_size + 1, /*step=*/1, options);
  torch::Tensor kv_cu_seq_lens = torch::range(
      /*start=*/0, /*end=*/max_batch_size + 1, /*step=*/1, options);
  torch::Tensor new_cache_slots = torch::zeros({max_batch_size}, options);

  const int64_t max_block_len = (k_max_seq_len + block_size - 1) / block_size;
  torch::Tensor block_tables =
      torch::zeros({max_batch_size, max_block_len}, options);

  LOG(INFO) << "Capturing CUDA graphs for batch sizes: " << batch_sizes;
  for (auto batch_size : batch_sizes) {
    auto graph = std::make_unique<CudaGraph>();

    // slice tensors for current batch size
    auto flatten_tokens =
        token_ids.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size);
    auto flatten_positions =
        positions.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size);

    InputParameters params;
    params.empty_kv_cache = false;
    params.num_sequences = batch_size;
    params.q_max_seq_len = 1;
    params.kv_max_seq_len = k_max_seq_len;
    params.q_cu_seq_lens = q_cu_seq_lens.slice(
        /*dim=*/0, /*start=*/0, /*end=*/batch_size + 1);
    params.kv_cu_seq_lens = kv_cu_seq_lens.slice(
        /*dim=*/0, /*start=*/0, /*end=*/batch_size + 1);
    params.new_cache_slots = new_cache_slots.slice(
        /*dim=*/0, /*start=*/0, /*end=*/batch_size);
    params.block_tables = block_tables.slice(
        /*dim=*/0, /*start=*/0, /*end=*/batch_size);

    // capture graph
    graph->capture(model_, flatten_tokens, flatten_positions, params, kv_cache);

    // save the graph
    graphs_[batch_size] = std::move(graph);
  }
  LOG(INFO) << "Finished capturing CUDA graphs";
}

// tokens: [num_tokens]
// positions: [num_tokens] token pos in the sequence
// returns: [num_tokens, hidden_size]
torch::Tensor ModelRunner::forward(const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   std::vector<KVCache>& kv_caches,
                                   const InputParameters& params) {
  const uint32_t batch_size = params.num_sequences;
  const uint32_t n_tokens = tokens.size(0);
  const uint32_t q_max_seq_len = params.q_max_seq_len;
  const uint32_t kv_max_seq_len = params.kv_max_seq_len;
  const bool use_kv_cache = !params.empty_kv_cache;

  // only use cuda graph for decoding batch when
  // 1> q_max_seq_len is 1
  // 2> max_seq_len is less than k_max_seq_len
  // 3> batch size in the list
  // ?? how to determine it is a decoding batch
  if (use_kv_cache && q_max_seq_len == 1 && kv_max_seq_len <= k_max_seq_len &&
      batch_size * q_max_seq_len == n_tokens) {
    auto it = graphs_.find(batch_size);
    if (it != graphs_.end()) {
      return it->second->replay(tokens, positions, params);
    }
  }

  return model_->forward(tokens, positions, kv_caches, params);
}

void ModelRunner::CudaGraph::capture(CausalLM* model,
                                     torch::Tensor flatten_tokens,
                                     torch::Tensor flatten_positions,
                                     const InputParameters& params,
                                     std::vector<KVCache>& kv_cache) {
  // run model once to avoid captured graph including initial benchmarking
  model->forward(flatten_tokens, flatten_positions, kv_cache, params);
  torch::cuda::synchronize();

  // save input tensors
  flatten_tokens_buffer_ = flatten_tokens;
  flatten_positions_buffer_ = flatten_positions;
  new_cache_slots_buffer_ = params.new_cache_slots;
  block_tables_buffer_ = params.block_tables;
  q_cu_seq_lens_buffer_ = params.q_cu_seq_lens;
  kv_cu_seq_lens_buffer_ = params.kv_cu_seq_lens;

  // create cuda graph and capture
  at::cuda::CUDAStream capture_stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard stream_guard(capture_stream);

  graph_ = std::make_unique<at::cuda::CUDAGraph>();
  graph_->capture_begin();
  auto hidden_states =
      model->forward(flatten_tokens, flatten_positions, kv_cache, params);
  graph_->capture_end();
  torch::cuda::synchronize();

  hidden_states_buffer_ = hidden_states;
}

torch::Tensor ModelRunner::CudaGraph::replay(torch::Tensor flatten_tokens,
                                             torch::Tensor flatten_positions,
                                             const InputParameters& params) {
  // TODO: use non_blocking copy
  // prepare input tensors
  flatten_tokens_buffer_.copy_(flatten_tokens, /*non_blocking=*/false);
  flatten_positions_buffer_.copy_(flatten_positions, /*non_blocking=*/false);
  q_cu_seq_lens_buffer_.copy_(params.q_cu_seq_lens, /*non_blocking=*/false);
  kv_cu_seq_lens_buffer_.copy_(params.kv_cu_seq_lens, /*non_blocking=*/false);
  new_cache_slots_buffer_.copy_(params.new_cache_slots, /*non_blocking=*/false);

  // it is possible that the block table with different padding length
  const int64_t block_table_len = params.block_tables.size(1);
  block_tables_buffer_.slice(/*dim=*/1, /*start=*/0, /*end=*/block_table_len)
      .copy_(params.block_tables, /*non_blocking=*/false);

  // replay the graph
  graph_->replay();

  // return the hidden states
  return hidden_states_buffer_;
}

}  // namespace llm
