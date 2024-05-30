#include "model_runner.h"

#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "common/metrics.h"
#include "memory/kv_cache.h"
#include "models/causal_lm.h"
#include "models/parameters.h"

DEFINE_COUNTER_FAMILY(num_model_execution_total,
                      "Total number of model execution");
DEFINE_COUNTER_INSTANCE(num_cuda_graph_replayed_total,
                        num_model_execution_total,
                        {{"mode", "cuda_graph"}});
DEFINE_COUNTER_INSTANCE(num_eager_execution_total,
                        num_model_execution_total,
                        {{"mode", "eager"}});

namespace llm {

// capture graph with batch size list
void ModelRunner::capture_cuda_graphs(std::vector<KVCache>& kv_cache) {
  if (!device_.is_cuda()) {
    // only capture CUDA graphs
    return;
  }
  if (options_.cuda_graph_batch_sizes().empty()) {
    // no batch sizes to capture CUDA graphs
    return;
  }

  // sort batch_sizes in descending order
  std::vector<uint32_t> sorted_batch_sizes = options_.cuda_graph_batch_sizes();
  std::sort(
      sorted_batch_sizes.begin(), sorted_batch_sizes.end(), std::greater<>());

  const int64_t num_decoding_tokens = options_.num_decoding_tokens();

  torch::DeviceGuard device_guard(device_);
  // allocate tensors for sharing among graphs
  auto options = torch::dtype(torch::kInt32).device(device_);
  const int64_t max_batch_size = sorted_batch_sizes.front();
  const int64_t max_num_tokens = num_decoding_tokens * max_batch_size;
  torch::Tensor token_ids = torch::zeros({max_num_tokens}, options);
  torch::Tensor positions = torch::zeros({max_num_tokens}, options);
  torch::Tensor q_cu_seq_lens = torch::range(
      /*start=*/0,
      /*end=*/max_num_tokens + 1,
      /*step=*/num_decoding_tokens,
      options);
  torch::Tensor kv_cu_seq_lens = torch::range(
      /*start=*/0,
      /*end=*/max_num_tokens + 1,
      /*step=*/num_decoding_tokens,
      options);
  torch::Tensor new_cache_slots = torch::zeros({max_num_tokens}, options);

  const auto max_seq_len = options_.cuda_graph_max_seq_len();
  const auto block_size = options_.block_size();
  // round up and add one additional block for speculative decoding
  const int64_t max_block_table_len =
      (max_seq_len + block_size - 1) / block_size + 1;
  torch::Tensor block_tables =
      torch::zeros({max_batch_size, max_block_table_len}, options);

  LOG(INFO) << "Capturing CUDA graphs on device: " << device_
            << ", num_decoding_tokens: " << num_decoding_tokens
            << ", batch sizes: " << sorted_batch_sizes;
  const auto shared_mem_pool = at::cuda::graph_pool_handle();
  for (auto batch_size : sorted_batch_sizes) {
    const int64_t n_tokens = num_decoding_tokens * batch_size;
    auto graph = std::make_unique<CudaGraph>();

    // slice tensors for current batch size
    auto flatten_tokens =
        token_ids.slice(/*dim=*/0, /*start=*/0, /*end=*/n_tokens);
    auto flatten_positions =
        positions.slice(/*dim=*/0, /*start=*/0, /*end=*/n_tokens);

    InputParameters params;
    params.empty_kv_cache = false;
    params.num_sequences = static_cast<int32_t>(batch_size);
    params.q_max_seq_len = static_cast<int32_t>(num_decoding_tokens);
    params.kv_max_seq_len = static_cast<int32_t>(max_seq_len);
    params.q_cu_seq_lens = q_cu_seq_lens.slice(
        /*dim=*/0, /*start=*/0, /*end=*/batch_size + 1);
    params.kv_cu_seq_lens = kv_cu_seq_lens.slice(
        /*dim=*/0, /*start=*/0, /*end=*/batch_size + 1);
    params.block_tables = block_tables.slice(
        /*dim=*/0, /*start=*/0, /*end=*/batch_size);
    params.new_cache_slots = new_cache_slots.slice(
        /*dim=*/0, /*start=*/0, /*end=*/n_tokens);

    // capture graph
    graph->capture(shared_mem_pool,
                   model_,
                   flatten_tokens,
                   flatten_positions,
                   params,
                   kv_cache);

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
  // check if captured graph exists
  auto it = graphs_.find(batch_size);
  if (it != graphs_.end()) {
    // kv_cache is not empty in decoding phase
    const bool in_decoding_phase = !params.empty_kv_cache;
    // max seq len is supported by captured graph
    const bool seq_len_supported =
        params.kv_max_seq_len <= options_.cuda_graph_max_seq_len();
    // each sequence has the same number of decoding tokens
    const uint32_t n_tokens = tokens.size(/*dim=*/0);
    const bool same_num_decoding_tokens =
        params.q_max_seq_len == options_.num_decoding_tokens() &&
        n_tokens == batch_size * options_.num_decoding_tokens();

    // replay the graph if all conditions are met
    if (in_decoding_phase && seq_len_supported && same_num_decoding_tokens) {
      COUNTER_INC(num_cuda_graph_replayed_total);
      return it->second->replay(tokens, positions, params);
    }
  }

  // run model directly in eager mode
  COUNTER_INC(num_eager_execution_total);
  return model_->forward(tokens, positions, kv_caches, params);
}

void ModelRunner::CudaGraph::capture(at::cuda::MempoolId_t mem_pool,
                                     CausalLM* model,
                                     torch::Tensor flatten_tokens,
                                     torch::Tensor flatten_positions,
                                     const InputParameters& params,
                                     std::vector<KVCache>& kv_cache) {
  CHECK(graph_ == nullptr) << "graph already captured";

  // save parameters
  batch_size_ = params.num_sequences;
  num_tokens_ = flatten_tokens.size(/*dim=*/0);

  // save input tensors
  flatten_tokens_ = flatten_tokens;
  flatten_positions_ = flatten_positions;
  new_cache_slots_ = params.new_cache_slots;
  block_tables_ = params.block_tables;
  q_cu_seq_lens_ = params.q_cu_seq_lens;
  kv_cu_seq_lens_ = params.kv_cu_seq_lens;

  // create cuda graph and capture
  at::cuda::CUDAStream capture_stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard stream_guard(capture_stream);

  // run model once to avoid captured graph including initial benchmarking
  model->forward(flatten_tokens, flatten_positions, kv_cache, params);
  capture_stream.synchronize();

  // capture the graph
  graph_ = std::make_unique<at::cuda::CUDAGraph>();
  graph_->capture_begin(mem_pool, cudaStreamCaptureModeThreadLocal);
  hidden_states_ =
      model->forward(flatten_tokens, flatten_positions, kv_cache, params);
  graph_->capture_end();
  capture_stream.synchronize();
}

torch::Tensor ModelRunner::CudaGraph::replay(torch::Tensor flatten_tokens,
                                             torch::Tensor flatten_positions,
                                             const InputParameters& params) {
  CHECK(graph_ != nullptr) << "graph not captured";

  const int64_t batch_size = params.num_sequences;
  const int64_t num_tokens = flatten_tokens.size(/*dim=*/0);
  const int64_t block_table_len = params.block_tables.size(/*dim=*/1);
  const int64_t max_block_table_len = block_tables_.size(/*dim=*/1);
  CHECK_EQ(batch_size, batch_size_) << "batch size mismatch";
  CHECK_EQ(num_tokens, num_tokens_) << "num tokens mismatch";
  CHECK_GE(max_block_table_len, block_table_len) << "block table size mismatch";

  // prepare input tensors
  flatten_tokens_.copy_(flatten_tokens, /*non_blocking=*/true);
  flatten_positions_.copy_(flatten_positions, /*non_blocking=*/true);
  q_cu_seq_lens_.copy_(params.q_cu_seq_lens, /*non_blocking=*/true);
  kv_cu_seq_lens_.copy_(params.kv_cu_seq_lens, /*non_blocking=*/true);
  new_cache_slots_.copy_(params.new_cache_slots, /*non_blocking=*/true);

  // it is possible that the block table with different padding length
  block_tables_.slice(/*dim=*/1, /*start=*/0, /*end=*/block_table_len)
      .copy_(params.block_tables, /*non_blocking=*/true);

  // replay the graph
  graph_->replay();

  // return the hidden states
  return hidden_states_;
}

}  // namespace llm
