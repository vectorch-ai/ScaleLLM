#include "response_handler.h"

#include <absl/synchronization/notification.h>
#include <absl/time/clock.h>
#include <glog/logging.h>

#include <memory>

#include "common/metrics.h"
#include "request/request.h"
#include "request/sequence.h"

// metrics

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

    // update the metrics for the request
    HISTOGRAM_OBSERVE(end_2_end_latency_seconds, request->elapsed_seconds());

    request->on_output(request->build_output(*tokenizer));
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
    if (seq.has_pending_tokens() || seq.is_finished()) {
      indexes.push_back(i);
      num_tokens.push_back(seq.num_tokens());
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
