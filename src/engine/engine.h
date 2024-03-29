#pragma once

#include "batch.h"
#include "models/model_args.h"
#include "quantization/quant_args.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_args.h"

namespace llm {

class BlockManager;
class Engine {
 public:
  virtual ~Engine() = default;

  // execute the model with the given batch, results are stored in the batch
  virtual void execute_model(Batch& batch) = 0;

  // validate multiple speculative tokens when use speculative decoding
  virtual void validate(Batch& batch) = 0;

  // return a clone of the tokenizer
  virtual std::unique_ptr<Tokenizer> tokenizer() const = 0;

  virtual BlockManager* block_manager() const = 0;

  virtual const ModelArgs& model_args() const = 0;

  virtual const QuantArgs& quant_args() const = 0;

  virtual const TokenizerArgs& tokenizer_args() const = 0;
};

}  // namespace llm
