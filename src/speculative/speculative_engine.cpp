#include "speculative_engine.h"

#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#include <memory>

#include "engine/llm_engine.h"
#include "engine/parameters.h"
#include "rejection_sampler.h"

namespace llm {

SpeculativeEngine::SpeculativeEngine(const Options& options)
    : options_(options) {
  CHECK_GT(options.num_speculative_tokens(), 0)
      << "speculative tokens should not be zero";

  // carry over the options
  LLMEngine::Options engine_options;
  engine_options.block_size(options.block_size())
      .max_cache_size(options.max_cache_size())
      .max_memory_utilization(options.max_memory_utilization())
      .enable_prefix_cache(options.enable_prefix_cache());
  // target engine
  engine_options.devices(options.devices());
  engine_options.num_decoding_tokens(options.num_speculative_tokens() + 1)
      .cuda_graph_max_seq_len(options.cuda_graph_max_seq_len())
      .cuda_graph_batch_sizes(options.cuda_graph_batch_sizes());
  engine_ = std::make_unique<LLMEngine>(engine_options);

  // draft engine
  engine_options.devices(options.draft_devices());
  engine_options.num_decoding_tokens(1)
      .cuda_graph_max_seq_len(options.cuda_graph_max_seq_len())
      .cuda_graph_batch_sizes(options.draft_cuda_graph_batch_sizes());
  draft_engine_ = std::make_unique<LLMEngine>(engine_options);

  // check if llm and ssm are using the same device
  for (const auto& target : options.devices()) {
    for (const auto& draft : options.draft_devices()) {
      if (target == draft) {
        share_device_ = true;
        break;
      }
    }
  }
}

bool SpeculativeEngine::init(const std::string& model_weights_path,
                             const std::string& draft_model_weights_path) {
  if (!init_model(model_weights_path, draft_model_weights_path)) {
    return false;
  }

  // init kv cache
  if (!init_kv_cache()) {
    return false;
  }

  // warmup the model
  if (!engine_->capture_cuda_graphs() ||
      !draft_engine_->capture_cuda_graphs()) {
    return false;
  }

  return true;
}

bool SpeculativeEngine::init_model(
    const std::string& model_weights_path,
    const std::string& draft_model_weights_path) {
  if (!engine_->init_model(model_weights_path)) {
    return false;
  }
  if (!draft_engine_->init_model(draft_model_weights_path)) {
    return false;
  }

  // TODO: check if the tokenizers are the same
  // TODO: check if the max context length are the same
  return true;
}

bool SpeculativeEngine::init_kv_cache() {
  // determine the kv cache size
  int64_t n_blocks = 0;
  const int64_t target_kv_cache_size = engine_->profile_memory_for_kv_cache();
  const int64_t draft_kv_cache_size =
      draft_engine_->profile_memory_for_kv_cache();

  // check if llm and ssm are using same device
  if (share_device_) {
    // on the same device, use the smaller kv cache size
    const int64_t kv_cache_size =
        std::min(target_kv_cache_size, draft_kv_cache_size);
    n_blocks = calculate_kv_cache_blocks(kv_cache_size);
  } else {
    // on different devices, use the smaller number of blocks
    const int64_t target_blocks =
        engine_->calculate_kv_cache_blocks(target_kv_cache_size);
    const int64_t draft_blocks =
        draft_engine_->calculate_kv_cache_blocks(draft_kv_cache_size);
    n_blocks = std::min(target_blocks, draft_blocks);
  }
  CHECK_GT(n_blocks, 0) << "no memory for kv cache";
  // init kv cache
  return engine_->init_kv_cache(n_blocks) &&
         draft_engine_->init_kv_cache(n_blocks);
}

ModelOutput SpeculativeEngine::execute_model(Batch& batch) {
  // run the draft model to get proposals
  std::vector<ModelOutput> draft_outputs;
  batch.set_engine_type(EngineType::SSM);
  for (size_t i = 0; i < options_.num_speculative_tokens(); ++i) {
    auto draft_output = draft_engine_->execute_model(batch);
    draft_outputs.push_back(draft_output);
  }

  // run the target model to get the verification scores
  batch.set_engine_type(EngineType::LLM);
  ModelOutput output = engine_->execute_model(batch);

  // verify the proposals with target and update the batch
  validate(batch, draft_outputs, output);
  return output;
}

void SpeculativeEngine::validate(Batch& batch,
                                 const std::vector<ModelOutput>& draft_outputs,
                                 const ModelOutput& target_output) {
  if (!target_output.sample_output.next_tokens.defined()) {
    // a pure prefill batch, no sampling needed
    return;
  }

  const auto bonus_token_ids =
      target_output.sample_output.next_tokens.view({-1, 1});
  const int64_t batch_size = bonus_token_ids.size(/*dim=*/0);
  const int64_t vocab_size = target_output.logits.size(/*dim=*/-1);
  const int64_t num_speculative_tokens =
      static_cast<int64_t>(draft_outputs.size());

  // [batch_size, n_speculative_tokens, vocab_size]
  auto target_logits = target_output.logits.view(
      {batch_size, num_speculative_tokens + /*bonus_tokens*/ 1, vocab_size});
  auto target_probs =
      torch::softmax(target_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  // filter out probs for bonus tokens
  target_probs = target_probs.slice(
      /*dim=*/1, /*start=*/0, /*end=*/num_speculative_tokens);

  // prepare input for rejection sampling
  std::vector<torch::Tensor> draft_token_ids_vec;
  std::vector<torch::Tensor> draft_probs_vec;
  for (const auto& draft_output : draft_outputs) {
    auto draft_token_ids =
        draft_output.sample_output.next_tokens.view({batch_size, 1});
    auto draft_probs =
        draft_output.sample_output.probs.view({{batch_size, 1, vocab_size}});
    draft_token_ids_vec.push_back(draft_token_ids);
    draft_probs_vec.push_back(draft_probs);
  }

  // concatenate the draft token ids and probs along the last dimension
  const auto draft_token_ids =
      torch::cat(draft_token_ids_vec, /*dim=*/1).to(bonus_token_ids);
  const auto draft_probs =
      torch::cat(draft_probs_vec, /*dim=*/1).to(target_probs);

  auto rejection_sampler =
      std::make_unique<RejectionSampler>(target_output.do_sample);

  // get the accepted tokens
  auto accepted_tokens = rejection_sampler->forward(
      draft_token_ids, draft_probs, target_probs, bonus_token_ids);

  // update the batch with the accpeted tokens
  batch.process_validate_output(accepted_tokens);
}

int64_t SpeculativeEngine::calculate_kv_cache_blocks(
    int64_t cache_size_in_bytes) const {
  CHECK_GT(cache_size_in_bytes, 0) << "no memory for kv cache";
  const int32_t block_size = options_.block_size();

  // compute the kv cache slot size in bytes
  const int64_t target_slot_size = engine_->kv_cache_slot_size_in_bytes();
  const int64_t draft_slot_size = draft_engine_->kv_cache_slot_size_in_bytes();

  // compute the number of blocks
  const int64_t block_size_in_bytes =
      block_size * (target_slot_size + draft_slot_size);
  return cache_size_in_bytes / block_size_in_bytes;
}

}  // namespace llm
