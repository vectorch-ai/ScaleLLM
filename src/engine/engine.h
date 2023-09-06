#pragma once

#include <ATen/core/TensorBody.h>

#include <memory>

#include "models/parameters.h"
#include "request/request.h"
#include "tokenizer/tokenizer.h"
#include "worker.h"

namespace llm {

// The Large Language Model (LLM) engine is a model runner designed to execute
// inference procedures incrementally using batches of requests. It comprises
// three critical components: a model, a tokenizer, and a resource manager.
// The inference process is primarily divided into two stages: 'prefill' and
// 'generate'.
// * 'Prefill': This is the more costly phase, as it involves processing a
// new prompt and generating the entire initial attention matrix.
// * 'Generate': In this phase, subsequent tokens are generated using the
// previously cached attention matrix.
// A single batch may contain requests from various stages of the inference
// process. The engine must be adept at handling these diverse requests,
// ensuring optimal resource management.

class Engine {
 public:
  // create an engine with the given devices
  Engine(const std::vector<torch::Device>& devices);

  virtual ~Engine() = default;

  bool init(const std::string& model_weights_path,
            const std::string& tokenizer_path);

  // step the engine forward by one step with the batch
  OutputParameters execute_model(const std::vector<Sequence*>& batch);

  const Tokenizer* tokenizer() const { return tokenizer_.get(); }

  const ModelArgs& model_args() const { return args_; }

 private:
  std::tuple<torch::Tensor, torch::Tensor, InputParameters, SamplingParameters>
  prepare_inputs(const std::vector<Request*>& batch);

  // a list of workers, with each worker handling a partial of model
  std::vector<std::unique_ptr<Worker>> workers_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;

  // model args
  ModelArgs args_;
};

}  // namespace llm
