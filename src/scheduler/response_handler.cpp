#include "response_handler.h"

#include <glog/logging.h>

#include <cstdint>
#include <memory>

#include "memory/block_manager.h"
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
  response_threadpool_.schedule([tokenizer = tokenizer_.get(),
                                 request = std::move(request)]() {
    RequestOutput req_output;
    // summarize statistics for all sequences
    Statistics& stats = req_output.stats;
    stats.num_prompt_tokens = request->num_prompt_tokens();
    for (const Sequence& seq : request->sequences) {
      stats.num_generated_tokens += seq.num_generated_tokens();
    }
    stats.num_total_tokens =
        stats.num_prompt_tokens + stats.num_generated_tokens;

    if (!request->is_streaming()) {
      auto& outputs = req_output.outputs;
      outputs.reserve(request->sequences.size());
      for (size_t i = 0; i < request->sequences.size(); ++i) {
        Sequence& seq = request->sequences[i];
        // generate the final output
        const auto output = seq.decode_delta_text(seq.token_ids(), *tokenizer);
        outputs.push_back({i, output, seq.finish_reason()});
      }
    }
    request->on_finish(Status(), req_output);
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
        req_output.outputs.push_back({index, std::move(delta), finish_reason});
      }
    }

    if (!request->on_stream(req_output)) {
      // cancel the request if on_stream returns false
      request->cancel();
    }
  });
}

}  // namespace llm
