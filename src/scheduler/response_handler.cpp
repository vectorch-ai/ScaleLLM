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

ResponseHandler::ResponseHandler(BlockManager* block_manager,
                                 Tokenizer* tokenizer)
    : block_manager_(block_manager), tokenizer_(tokenizer) {}

void ResponseHandler::on_request_finish(std::unique_ptr<Request> request) {
  // release all blocks for the finished request
  block_manager_->release_blocks_for(request.get());
  // schedule the response handling
  response_threadpool_.schedule([tokenizer = tokenizer_,
                                 request = std::move(request)]() {
    if (request->stream) {
      // just finish the request
      request->on_stream_finish(Status());
    } else {
      // summarize statistics for all sequences
      Statistics stats;
      stats.num_prompt_tokens = request->num_prompt_tokens();
      for (const Sequence& seq : request->sequences) {
        stats.num_generated_tokens += seq.num_generated_tokens();
      }
      stats.num_total_tokens =
          stats.num_prompt_tokens + stats.num_generated_tokens;

      std::vector<SequenceOutput> seq_results;
      seq_results.reserve(request->sequences.size());
      for (Sequence& seq : request->sequences) {
        // generate the final output
        const auto output = seq.decode_delta_text(seq.token_ids(), *tokenizer);
        seq_results.push_back({output, seq.finish_reason()});
      }
      request->on_finish(seq_results, Status(), stats);
    }
  });
}

void ResponseHandler::on_sequence_stream(Sequence* seq) {
  // check if the sequence has enough tokens to output
  const auto token_ids = seq->token_ids();
  const size_t output_offset = seq->output_offset();
  const size_t num_tokens_to_output = token_ids.size() - output_offset;
  if (seq->is_finished() ||
      num_tokens_to_output >= FLAGS_streaming_token_buffer_size) {
    const auto finish_reason = seq->finish_reason();
    // output the delta text til the end of the sequence to the client
    response_threadpool_.schedule(
        [seq, tokenizer = tokenizer_, token_ids = token_ids, finish_reason]() {
          auto delta = seq->decode_delta_text(token_ids, *tokenizer);
          if (!delta.empty() || finish_reason != FinishReason::NONE) {
            seq->stream_delta({std::move(delta), finish_reason});
          };
        });
  }
}

}  // namespace llm
