#pragma once

#include <common/threadpool.h>

#include <cstdint>

namespace llm {

class BlockManager;
class Request;
class Sequence;
class Tokenizer;
class ResponseHandler final {
 public:
  ResponseHandler(std::unique_ptr<Tokenizer> tokenizer);

  // take over the ownership of the request
  void on_request_finish(std::unique_ptr<Request> request);

  void on_request_stream(Request* request);

  // wait for all responses in queue to be handled
  void wait_for_complete();

 private:
  // the threadpool to handle responses
  ThreadPool response_threadpool_;

  // tokenizer instance to decode token ids
  std::unique_ptr<Tokenizer> tokenizer_;
};

}  // namespace llm
