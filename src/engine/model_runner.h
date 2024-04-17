#pragma once

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <memory>

#include "memory/kv_cache.h"
#include "models/causal_lm.h"
#include "models/parameters.h"

namespace llm {

class ModelRunner final {
 public:
  ModelRunner(CausalLM* model) : model_(model) {}

  // capture graph with batch size list
  void capture_graphs(const std::vector<uint32_t>& batch_size_list,
                      std::vector<KVCache>& kv_cache);

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& parameters);

 private:
  // model, do not own
  CausalLM* model_;

  // number of decoding tokens per sequence
  int64_t num_decoding_tokens_ = 1;

  class CudaGraph;
  // captured cuda graphs, mapping from batch size to cudagraph
  std::unordered_map<int32_t, std::unique_ptr<CudaGraph>> graphs_;

  class CudaGraph final {
   public:
    void capture(CausalLM* model,
                 torch::Tensor flatten_tokens,
                 torch::Tensor flatten_positions,
                 const InputParameters& params,
                 std::vector<KVCache>& kv_cache);

    torch::Tensor replay(torch::Tensor flatten_tokens,
                         torch::Tensor flatten_positions,
                         const InputParameters& params,
                         const std::vector<KVCache>& /*kv_cache*/);

   private:
    std::unique_ptr<at::cuda::CUDAGraph> graph_;
    // input tensors
    torch::Tensor flatten_tokens_buffer_;
    torch::Tensor flatten_positions_buffer_;
    torch::Tensor new_cache_slots_buffer_;
    torch::Tensor block_tables_buffer_;
    torch::Tensor q_cu_seq_lens_buffer_;
    torch::Tensor kv_cu_seq_lens_buffer_;

    // output tensors
    torch::Tensor hidden_states_buffer_;
  };
};

}  // namespace llm
