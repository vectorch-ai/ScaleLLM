#pragma once

#include <common/threadpool.h>
#include <cstdint>

namespace llm {

class BlockManager;
class Request;
class Sequence;
class Tokenizer;
class ResponseHandler {
 public:
  ResponseHandler(BlockManager* block_manager, Tokenizer* tokenizer);
  virtual ~ResponseHandler() = default;

  virtual void on_request_finish(Request* request);
  virtual void on_sequence_stream(Sequence* seq);

 private:
  // the threadpool to handle responses
  ThreadPool response_threadpool_;

  BlockManager* block_manager_;
  Tokenizer* tokenizer_;
};

}  // namespace llm
