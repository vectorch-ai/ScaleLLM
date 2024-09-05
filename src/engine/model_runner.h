#pragma once

#include <ATen/cuda/CUDAGraph.h>
#include <absl/container/flat_hash_map.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "memory/kv_cache.h"
#include "models/causal_lm.h"
#include "models/parameters.h"

namespace llm {

class ModelRunner final {
 public:
  struct Options {
    // number of decoding tokens per sequence
    // in speculative decoding, it is the number of speculative tokens + 1
    DEFINE_ARG(int64_t, num_decoding_tokens) = 1;

    // number of slots per block
    DEFINE_ARG(int64_t, block_size) = 16;

    // max sequence length used to capture cuda graphs
    DEFINE_ARG(int64_t, cuda_graph_max_seq_len) = 1024;

    // batch sizes to capture cuda graphs
    DEFINE_ARG(std::vector<uint32_t>, cuda_graph_batch_sizes);
  };

  ModelRunner(CausalLM* model,
              const torch::Device& device,
              const Options& options);

  // capture graph for given batch size
  void capture_cuda_graphs(uint32_t batch_size, std::vector<KVCache>& kv_cache);

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  // returns: [num_tokens, hidden_size]
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& params);

 private:
  // model, do not own
  CausalLM* model_;

  // device to run the model
  torch::Device device_;

  // options
  Options options_;

  // shared inputs and outputs for the model
  uint32_t max_batch_size_ = 0;
  torch::Tensor token_ids_;
  torch::Tensor positions_;
  torch::Tensor q_cu_seq_lens_;
  torch::Tensor kv_cu_seq_lens_;
  torch::Tensor new_cache_slots_;
  torch::Tensor block_tables_;
  torch::Tensor cu_block_lens_;

  // graph pool handler
  at::cuda::MempoolId_t mem_pool_;

  class CudaGraph;
  // captured cuda graphs, mapping from batch size to graph
  absl::flat_hash_map<uint32_t, std::unique_ptr<CudaGraph>> graphs_;

  class CudaGraph final {
   public:
    void capture(at::cuda::MempoolId_t mem_pool,
                 CausalLM* model,
                 torch::Tensor flatten_tokens,
                 torch::Tensor flatten_positions,
                 const InputParameters& params,
                 std::vector<KVCache>& kv_cache);

    torch::Tensor replay(torch::Tensor flatten_tokens,
                         torch::Tensor flatten_positions,
                         const InputParameters& params);

   private:
    std::unique_ptr<at::cuda::CUDAGraph> graph_;

    // batch size that
    int64_t batch_size_ = 0;
    // max number of tokens that can be processed by the captured graph
    int64_t num_tokens_ = 0;

    // input tensors
    torch::Tensor flatten_tokens_;
    torch::Tensor flatten_positions_;
    torch::Tensor new_cache_slots_;
    torch::Tensor block_tables_;
    torch::Tensor cu_block_lens_;
    torch::Tensor q_cu_seq_lens_;
    torch::Tensor kv_cu_seq_lens_;

    // output tensors
    torch::Tensor hidden_states_;
  };
};

}  // namespace llm
