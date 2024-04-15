#include "engine/worker.h"

#include <gtest/gtest.h>

#include "engine/batch.h"
#include "memory/block_manager.h"
#include "models/simple_model.h"
#include "quantization/quant_args.h"

namespace llm {

class TestableWorker {
 public:
  TestableWorker(const torch::Device& device) : device_(device) {
    worker_ = std::make_unique<Worker>(ParallelArgs(0, 1, nullptr), device_);
    dtype_ = torch::kHalf;
  }

  // Three steps to initialize worker:
  // 1. init_model, 2. init_kv_cache, 3. warmup_model
  bool init(bool enable_cudagraph) {
    if (!init_model()) {
      return false;
    }

    if (!init_kv_cache()) {
      return false;
    }

    if (!warmup_model(enable_cudagraph)) {
      return false;
    }
    return true;
  }

  ModelOutput execute_model(const ModelInput& inputs) {
    return worker_->execute_model(inputs);
  }

  bool init_model() {
    init_model_args();

    if (!worker_->init_model(dtype_, args_, quant_args_)) {
      return false;
    }

    auto state_dict = create_state_dict();
    worker_->load_state_dict(state_dict);
    worker_->verify_loaded_weights();
    return true;
  }

  bool warmup_model(bool enable_cudagraph) {
    return worker_->warmup_model(enable_cudagraph);
  }

  bool init_kv_cache() {
    const int64_t cache_size_in_bytes = 1024 * 1024 * 1024;
    const int64_t block_size = 256;
    const int64_t n_heads = args_.n_heads();
    const int64_t head_dim = args_.hidden_size() / n_heads;
    const auto dtype_size = torch::scalarTypeToTypeMeta(dtype_).itemsize();
    const int64_t block_size_in_bytes =
        2 * block_size * n_heads * head_dim * args_.n_layers() * dtype_size;

    const int64_t n_blocks = cache_size_in_bytes / block_size_in_bytes;
    const std::vector<int64_t> kv_cache_shape = {
        n_blocks, block_size, n_heads, head_dim};

    BlockManager::Options options;
    options.num_blocks(n_blocks).block_size(block_size);
    block_manager_ = std::make_unique<BlockManager>(options);
    return worker_->init_kv_cache(kv_cache_shape);
  }

 private:
  void init_model_args() {
    // SimpleModel's ModelArgs
    args_.model_type("simple");
    args_.vocab_size(32000);
    args_.hidden_size(4096);
    args_.n_layers(1);
    args_.n_heads(32);
    args_.n_kv_heads(32);
    args_.intermediate_size(11008);
    args_.hidden_act("silu");
    args_.max_position_embeddings(2048);
  }

  StateDict create_state_dict() {
    // SimpleModel's StateDict, tensor shape based on ModelArgs
    std::unordered_map<std::string, torch::Tensor> dict;
    // model.embed_tokens.weight: [32000, 4096]
    dict.emplace("model.embed_tokens.weight",
                 torch::ones({32000, 4096}, dtype_));
    // torch::ones({32000, 4096}, torch::kHalf, device_));
    // model.layers.0.mlp.down_proj.weight: [4096, 11008]
    dict.emplace("model.layers.0.mlp.down_proj.weight",
                 torch::ones({4096, 11008}, dtype_));
    // torch::ones({4096, 11008}, torch::kHalf, device_));
    // model.layers.0.mlp.gate_proj.weight: [11008, 4096]
    dict.emplace("model.layers.0.mlp.gate_proj.weight",
                 torch::ones({11008, 4096}, dtype_));
    // torch::ones({11008, 4096}, torch::kHalf, device_));
    // model.layers.0.mlp.up_proj.weight: [11008, 4096]
    dict.emplace("model.layers.0.mlp.up_proj.weight",
                 torch::ones({11008, 4096}, dtype_));
    // torch::ones({11008, 4096}, torch::kHalf, device_));
    StateDict state_dict(dict, 0, 1);
    return state_dict;
  }

 private:
  torch::Device device_;
  ModelArgs args_;
  QuantArgs quant_args_;
  torch::ScalarType dtype_;

  std::unique_ptr<Worker> worker_;
  std::unique_ptr<BlockManager> block_manager_;
};

// Test pass.
TEST(WorkerTest, InitSimpleModelBasic) {
  torch::Device device(torch::kCUDA);
  TestableWorker worker(device);
  EXPECT_TRUE(worker.init_model());
  EXPECT_TRUE(worker.init_kv_cache());
  // disable cudagraph
  EXPECT_TRUE(worker.warmup_model(false));
}

// TODO: add branch to separate prefill & decode
TEST(WorkerTest, ExecuteSimpleModelBasic) {
  /*
  torch::Device device(torch::kCUDA);
  TestableWorker worker(device);
  // disable cudagraph
  EXPECT_TRUE(worker.init(false));
  torch::Tensor flatten_tokens;
  torch::Tensor flatten_positions;
  InputParameters params;
  SamplingParameters sampling_params;
  worker.execute_model(flatten_tokens, flatten_positions, params,
      sampling_params);*/
}

// TODO: fix coredump
TEST(WorkerTest, InitSimpleModelWithCudaGraph) {
  /*
  torch::Device device(torch::kCUDA);
  TestableWorker worker(device);
  EXPECT_TRUE(worker.init_model());
  EXPECT_TRUE(worker.init_kv_cache());
  bool enable_cudagraph = true;
  EXPECT_TRUE(worker.warmup_model(enable_cudagraph));*/
}

// TODO: fix coredump
TEST(WorkerTest, ExecuteSimpleModelWithCudaGraph) {
  /*
  torch::Device device(torch::kCUDA);
  TestableWorker worker(device);
  // enable cudagraph
  EXPECT_TRUE(worker.init(true));
  torch::Tensor flatten_tokens;
  torch::Tensor flatten_positions;
  InputParameters params;
  SamplingParameters sampling_params;
  worker.execute_model(flatten_tokens, flatten_positions, params,
      sampling_params);*/
}

}  // namespace llm
