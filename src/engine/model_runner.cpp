#include "model_runner.h"

#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "memory/kv_cache.h"
#include "models/causal_lm.h"
#include "models/parameters.h"

namespace llm {

const static std::vector<int> BatchSizeForCudaGraph = {
    1,   2,   4,   8,   16,  24,  32,  40,  48,  56,  64,  72,
    80,  88,  96,  104, 112, 120, 128, 136, 144, 152, 160, 168,
    176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256};
constexpr int MaxBatchSizeForCudaGraph = 256;
constexpr int64_t max_seq_len = 1024;

// capture graph with batch size list
void ModelRunner::capture_graphs(const std::vector<uint32_t>& batch_size_list,
                                 std::vector<KVCache>& kv_cache) {
  // allocate memory for inputs and outputs
  torch::Tensor flatten_tokens;
  torch::Tensor flatten_positions;
  InputParameters input_params;
  // construct input parameters

  // for (auto batch_size : batch_size_list) {
  //   auto graph = std::make_unique<at::cuda::CUDAGraph>();
  //   graph->capture_begin();
  //   auto hidden_states = model_->forward(
  //       flatten_tokens, flatten_positions, kv_cache, input_params);
  //   graph->capture_end();
  //   torch::cuda::synchronize();
  //   graphs_[batch_size] = std::move(graph);
  // }
}

// tokens: [num_tokens]
// positions: [num_tokens] token pos in the sequence
// returns: [num_tokens, hidden_size]
torch::Tensor ModelRunner::forward(const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   std::vector<KVCache>& kv_caches,
                                   const InputParameters& parameters) {
  // bool has_graph = true;
  // if (has_graph) {
  //   // auto batch_size = tokens.size(0);
  //   // auto* graph = graphs_[batch_size].get();
  //   // return replay(tokens, positions, parameters, batch_size);
  // }

  return model_->forward(tokens, positions, kv_caches, parameters);
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

  // create cudagraph and capture
  {
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
}

torch::Tensor ModelRunner::CudaGraph::replay(
    torch::Tensor flatten_tokens,
    torch::Tensor flatten_positions,
    const InputParameters& params,
    const std::vector<KVCache>& /*kv_cache*/) {
  // prepare input tensors
  flatten_tokens_buffer_.copy_(flatten_tokens, /*non_blocking=*/false);
  flatten_positions_buffer_.copy_(flatten_positions, /*non_blocking=*/false);
  new_cache_slots_buffer_.copy_(params.new_cache_slots, /*non_blocking=*/false);
  block_tables_buffer_.copy_(params.block_tables, /*non_blocking=*/false);
  q_cu_seq_lens_buffer_.copy_(params.q_cu_seq_lens, /*non_blocking=*/false);
  kv_cu_seq_lens_buffer_.copy_(params.kv_cu_seq_lens, /*non_blocking=*/false);

  // replay the graph
  graph_->replay();

  // return the hidden states
  return hidden_states_buffer_;
}

}  // namespace llm
