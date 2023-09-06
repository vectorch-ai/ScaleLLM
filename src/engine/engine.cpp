#include "engine.h"

#include <memory>

#include "models/model_loader.h"
#include "request/request.h"
#include "tokenizer/sentencepiece_tokenizer.h"
#include "utils.h"
#include "worker.h"

DECLARE_int32(max_seq_len);
DECLARE_int32(max_batch_size);

namespace llm {

Engine::Engine(const std::vector<torch::Device>& devices) {
  // create a worker for each device
  for (const auto& device : devices) {
    workers_.emplace_back(std::make_unique<Worker>(device));
  }
}

bool Engine::init(const std::string& model_weights_path,
                  const std::string& tokenizer_path) {
  // load tokenizer
  // TODO: support other tokenizers
  tokenizer_ = std::make_unique<SentencePieceTokenizer>(tokenizer_path);

  ModelLoader model_loader(model_weights_path);
  CHECK_EQ(workers_.size(), model_loader.weights_files_count())
      << "The number of workers should be the same as the number of model "
         "weights files";

  args_ = model_loader.model_args();
  if (args_.vocab_size() == -1) {
    args_.vocab_size(static_cast<int64_t>(tokenizer_->vocab_size()));
  }
  // TODO: remove this two from model args
  args_.max_seq_len(FLAGS_max_seq_len).max_batch_size(FLAGS_max_batch_size);


  // init workers
  if (workers_.size() == 1) {
    // only one worker, call blocking init
    if (!workers_[0]->init(args_)) {
      return false;
    }
  } else {
    // multiple workers, call async init
    std::vector<folly::SemiFuture<bool>> futures;
    futures.reserve(workers_.size());
    for (auto& worker : workers_) {
      futures.push_back(worker->init_async(args_));
    }
    // wait for all futures to complete
    auto results = folly::collectAll(futures).get();
    for (const auto& result : results) {
      if (!result.value()) {
        return false;
      }
    }
  }

  // load the weights from the checkpoint
  // each worker loads one model weights file
  // TODO: add support for loading multiple model weights files for each worker
  size_t i = 0;
  for (const auto& state_dict : model_loader) {
    workers_[i++]->load_state_dict(state_dict);
  }
  return true;
}

OutputParameters Engine::execute_model(const std::vector<Request*>& batch) {
  // prepare inputs for workers
  torch::Tensor input_token_ids;
  torch::Tensor input_positions;
  InputParameters input_params;
  SamplingParameters sampling_params;
  Utils::prepare_inputs(batch,
                        &input_token_ids,
                        &input_positions,
                        &input_params,
                        &sampling_params);
  if (workers_.size() == 1) {
    // only one worker, call blocking forward
    return workers_[0]->execute_model(
        input_token_ids, input_positions, input_params, sampling_params);
  }
  
  std::vector<folly::SemiFuture<OutputParameters>> futures;
  futures.reserve(workers_.size());
  for (auto& worker : workers_) {
    futures.push_back(worker->execute_model_async(
        input_token_ids, input_positions, input_params, sampling_params));
  }
  // wait for the all future to complete
  auto results = folly::collectAll(futures).get();
  // return the result from the first worker
  return results[0].value();
  //TODO: mapping back to the original request in the batch
  
}

}  // namespace llm
