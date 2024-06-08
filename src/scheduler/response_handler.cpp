#include "response_handler.h"

#include <absl/synchronization/notification.h>
#include <absl/time/clock.h>
#include <glog/logging.h>

#include <memory>

#include "common/metrics.h"
#include "request/request.h"
#include "request/sequence.h"
#include "request/status.h"

// metrics
DEFINE_COUNTER_FAMILY(detokenization_latency_seconds,
                      "Latency of detokenization in seconds");
DEFINE_COUNTER_INSTANCE(stream_decode_latency_seconds,
                        detokenization_latency_seconds,
                        {{"mode", "stream"}});
DEFINE_COUNTER_INSTANCE(non_stream_decode_latency_seconds,
                        detokenization_latency_seconds,
                        {{"mode", "non-stream"}});

DEFINE_COUNTER_FAMILY(responsing_latency_seconds,
                      "Latency of responding in seconds");
DEFINE_COUNTER_INSTANCE(stream_responsing_latency_seconds,
                        responsing_latency_seconds,
                        {{"mode", "stream"}});
DEFINE_COUNTER_INSTANCE(non_stream_responsing_latency_seconds,
                        responsing_latency_seconds,
                        {{"mode", "non-stream"}});

DEFINE_HISTOGRAM(
    end_2_end_latency_seconds,
    "Histogram of end to end latency in seconds",
    std::vector<double>{0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0});

namespace llm {

ResponseHandler::ResponseHandler(const Tokenizer* tokenizer)
    : tokenizer_(tokenizer->clone()) {}

void ResponseHandler::on_request_finish(std::unique_ptr<Request> request) {
  // schedule the response handling
  response_threadpool_.schedule([tokenizer = tokenizer_.get(),
                                 request = std::move(request)]() {
    AUTO_COUNTER(non_stream_responsing_latency_seconds);

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
    HISTOGRAM_OBSERVE(end_2_end_latency_seconds, request->elapsed_seconds());

    if (!request->is_streaming()) {
      auto& outputs = req_output.outputs;
      outputs.reserve(request->sequences.size());
      for (auto& seq : request->sequences) {
        // generate the final output
        AUTO_COUNTER(non_stream_decode_latency_seconds);
        auto seq_output = seq.build_output(*tokenizer);
        if (seq_output.has_value()) {
          outputs.push_back(std::move(seq_output.value()));
        }
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
  std::vector<size_t> num_tokens;
  for (size_t i = 0; i < request->sequences.size(); ++i) {
    Sequence& seq = request->sequences[i];
    if (seq.is_closed()) {
      // skip already closed sequences
      continue;
    }

    // check if the sequence has enough tokens to output
    const auto size = seq.num_tokens();
    if (seq.is_finished() || size > seq.output_offset()) {
      indexes.push_back(i);
      num_tokens.push_back(size);
    }

    // close the sequence after sending finish reason
    if (seq.is_finished()) {
      seq.close();
    }
  }

  // output the delta text til the end of the sequence to the client
  response_threadpool_.schedule([request,
                                 indexes = std::move(indexes),
                                 num_tokens = std::move(num_tokens),
                                 tokenizer = tokenizer_.get()]() {
    AUTO_COUNTER(stream_responsing_latency_seconds);

    RequestOutput req_output;
    for (size_t i = 0; i < indexes.size(); ++i) {
      const size_t index = indexes[i];
      const size_t size = num_tokens[i];
      Sequence& seq = request->sequences[index];

      AUTO_COUNTER(stream_decode_latency_seconds);
      auto seq_output = seq.build_delta_output_until(size, *tokenizer);
      if (seq_output.has_value()) {
        req_output.outputs.push_back(std::move(seq_output.value()));
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
