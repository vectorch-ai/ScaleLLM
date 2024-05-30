#include "response_handler.h"

#include <absl/synchronization/notification.h>
#include <absl/time/clock.h>
#include <glog/logging.h>

#include <cstdint>
#include <memory>

#include "common/metrics.h"
#include "request/request.h"
#include "request/sequence.h"
#include "request/status.h"

// gflags
DEFINE_int32(streaming_token_buffer_size,
             1,
             "number of tokens to buffer before streaming to client");

// metrics
DEFINE_COUNTER(prompt_tokens_total, "Total number of prompt tokens");
DEFINE_COUNTER(generated_tokens_total, "Total number of generated tokens");

DEFINE_HISTOGRAM(
    end_2_end_latency_seconds,
    "Histogram of end to end latency in seconds",
    std::vector<double>{0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0});

// ttft latency histogram
// DEFINE_HISTOGRAM(
//     time_to_first_token_latency_seconds,
//     "Histogram of time to first token latency in seconds",
//     std::vector<double>{0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0});
// inter token latency histogram
// DEFINE_HISTOGRAM(
//     inter_token_latency_seconds,
//     "Histogram of inter token latency in seconds",
//     std::vector<double>{0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0});

namespace llm {

ResponseHandler::ResponseHandler(const Tokenizer* tokenizer)
    : tokenizer_(tokenizer->clone()) {}

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

        // update the metrics for the request
        COUNTER_ADD(prompt_tokens_total, usage.num_prompt_tokens);
        COUNTER_ADD(generated_tokens_total, usage.num_generated_tokens);

        const auto duration = absl::Now() - request->created_time;
        HISTOGRAM_OBSERVE(end_2_end_latency_seconds,
                          absl::ToDoubleSeconds(duration));

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
        req_output.status = Status(StatusCode::OK);
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
    if (seq.is_closed()) {
      // skip already closed sequences
      continue;
    }

    // check if the sequence has enough tokens to output
    const auto ids = seq.token_ids();
    if (seq.is_finished() ||
        ids.size() - seq.output_offset() >= FLAGS_streaming_token_buffer_size) {
      indexes.push_back(i);
      token_ids.push_back(ids);
    }

    // close the sequence after sending finish reason
    if (seq.is_finished()) {
      seq.close();
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
