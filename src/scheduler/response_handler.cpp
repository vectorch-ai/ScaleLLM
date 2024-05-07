#include "response_handler.h"

#include <absl/synchronization/notification.h>
#include <glog/logging.h>

#include <cstdint>
#include <memory>

#include "request/request.h"
#include "request/sequence.h"

namespace llm {

DEFINE_int32(streaming_token_buffer_size,
             1,
             "number of tokens to buffer before streaming to client");

ResponseHandler::ResponseHandler(std::unique_ptr<Tokenizer> tokenizer)
    : tokenizer_(std::move(tokenizer)) {}

void ResponseHandler::on_request_finish(std::unique_ptr<Request> request) {
  // schedule the response handling
  response_threadpool_.schedule(
      [tokenizer = tokenizer_.get(), request = std::move(request)]() {
        RequestOutput req_output;
        // summarize statistics for all sequences
        Usage usage;
        usage.num_prompt_tokens = request->num_prompt_tokens();
        for (const Sequence& seq : request->sequences) {
          usage.num_generated_tokens += seq.num_generated_tokens();
        }
        usage.num_total_tokens =
            usage.num_prompt_tokens + usage.num_generated_tokens;
        req_output.usage = usage;

        if (!request->is_streaming()) {
          auto& outputs = req_output.outputs;
          outputs.reserve(request->sequences.size());
          for (size_t i = 0; i < request->sequences.size(); ++i) {
            Sequence& seq = request->sequences[i];
            const auto finish_reason = seq.finish_reason();
            // generate the final output
            auto output = seq.decode_delta_text(seq.token_ids(), *tokenizer);
            outputs.push_back({i, std::move(output), to_string(finish_reason)});
          }
        }
        req_output.finished = true;
        request->on_output(req_output);
      });
}

void ResponseHandler::on_request_stream(Request* request) {
  CHECK(request->is_streaming()) << "request is not a streaming request";

  std::vector<size_t> indexes;
  std::vector<Slice<int32_t>> token_ids;
  for (size_t i = 0; i < request->sequences.size(); ++i) {
    Sequence& seq = request->sequences[i];
    if (seq.num_blocks() == 0) {
      CHECK(seq.is_finished());
      // skip already finished sequences
      continue;
    }

    // check if the sequence has enough tokens to output
    const auto ids = seq.token_ids();
    if (seq.is_finished() ||
        ids.size() - seq.output_offset() >= FLAGS_streaming_token_buffer_size) {
      indexes.push_back(i);
      token_ids.push_back(ids);
    }
  }

  // output the delta text til the end of the sequence to the client
  response_threadpool_.schedule([request,
                                 indexes = std::move(indexes),
                                 token_ids = std::move(token_ids),
                                 tokenizer = tokenizer_.get()]() {
    RequestOutput req_output;
    for (size_t i = 0; i < indexes.size(); ++i) {
      const size_t index = indexes[i];
      Sequence& seq = request->sequences[index];
      const auto finish_reason = seq.finish_reason();
      auto delta = seq.decode_delta_text(token_ids[i], *tokenizer);
      if (!delta.empty() || finish_reason != FinishReason::NONE) {
        req_output.outputs.push_back(
            {index, std::move(delta), to_string(finish_reason)});
      }
    }

    if (!request->on_output(req_output)) {
      // cancel the request if on_stream returns false
      request->cancel();
    }
  });
}

void ResponseHandler::wait_for_complete() {
  // add a task to the end of the pool to wait for it to finish
  absl::Notification done;
  response_threadpool_.schedule([&done]() { done.Notify(); });
  done.WaitForNotification();
}

}  // namespace llm
