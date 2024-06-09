#include <folly/init/Init.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/metrics.h"
#include "handlers/llm_handler.h"
#include "handlers/sampling_params.h"
#include "request/output.h"
#include "request/status.h"

namespace llm::csrc {
namespace py = pybind11;

// NOLINTNEXTLINE
static std::string get_metrics() { return Metrics::Instance().GetString(); }

PYBIND11_MODULE(PY_MODULE_NAME, m) {
  // glog and glfag will be initialized in folly::init
  //   int argc = 0;
  //   char** argv = nullptr;
  //   folly::Init init(&argc, &argv);
  m.def("get_metrics", &get_metrics);

  // class SamplingParameter
  py::class_<SamplingParams>(m, "SamplingParams")
      .def(py::init<uint32_t, /*max_tokens*/
                    uint32_t, /*n*/
                    uint32_t, /*best_of*/
                    bool,     /*echo*/
                    float,    /*frequency_penalty*/
                    float,    /*presence_penalty*/
                    float,    /*repetition_penalty*/
                    float,    /*temperature*/
                    float,    /*top_p*/
                    int64_t,  /*top_k*/
                    bool,     /*logprobs*/
                    int64_t,  /*top_logprobs*/
                    bool,     /*skip_special_tokens*/
                    bool,     /*ignore_eos*/
                    std::optional<std::vector<std::string>>, /*stop*/
                    std::optional<std::vector<int32_t>>>(),  /*stop_token_ids*/
           py::arg("max_tokens") = 16,
           py::arg("n") = 1,
           py::arg("best_of") = 1,
           py::arg("echo") = false,
           py::arg("frequency_penalty") = 0.0,
           py::arg("presence_penalty") = 0.0,
           py::arg("repetition_penalty") = 1.0,
           py::arg("temperature") = 1.0,
           py::arg("top_p") = 1.0,
           py::arg("top_k") = -1,
           py::arg("logprobs") = false,
           py::arg("top_logprobs") = 0,
           py::arg("skip_special_tokens") = true,
           py::arg("ignore_eos") = false,
           py::arg("stop") = std::nullopt,
           py::arg("stop_token_ids") = std::nullopt)
      .def_readwrite("max_tokens", &SamplingParams::max_tokens)
      .def_readwrite("n", &SamplingParams::n)
      .def_readwrite("best_of", &SamplingParams::best_of)
      .def_readwrite("echo", &SamplingParams::echo)
      .def_readwrite("frequency_penalty", &SamplingParams::frequency_penalty)
      .def_readwrite("presence_penalty", &SamplingParams::presence_penalty)
      .def_readwrite("repetition_penalty", &SamplingParams::repetition_penalty)
      .def_readwrite("temperature", &SamplingParams::temperature)
      .def_readwrite("top_p", &SamplingParams::top_p)
      .def_readwrite("top_k", &SamplingParams::top_k)
      .def_readwrite("logprobs", &SamplingParams::logprobs)
      .def_readwrite("top_logprobs", &SamplingParams::top_logprobs)
      .def_readwrite("skip_special_tokens",
                     &SamplingParams::skip_special_tokens)
      .def_readwrite("ignore_eos", &SamplingParams::ignore_eos)
      .def_readwrite("stop", &SamplingParams::stop)
      .def_readwrite("stop_token_ids", &SamplingParams::stop_token_ids);

  py::class_<Message>(m, "Message")
      .def(py::init<const std::string&, const std::string&>(),
           py::arg("role"),
           py::arg("content"))
      .def_readwrite("role", &Message::role)
      .def_readwrite("content", &Message::content);

  py::enum_<Priority>(m, "Priority")
      .value("DEFAULT", Priority::NORMAL)
      .value("LOW", Priority::LOW)
      .value("NORMAL", Priority::NORMAL)
      .value("HIGH", Priority::HIGH)
      .export_values();

  py::class_<Usage>(m, "Usage")
      .def(py::init())
      .def_readwrite("num_prompt_tokens", &Usage::num_prompt_tokens)
      .def_readwrite("num_generated_tokens", &Usage::num_generated_tokens)
      .def_readwrite("num_total_tokens", &Usage::num_total_tokens);

  py::enum_<StatusCode>(m, "StatusCode")
      .value("OK", StatusCode::OK)
      .value("CANCELLED", StatusCode::CANCELLED)
      .value("UNKNOWN", StatusCode::UNKNOWN)
      .value("INVALID_ARGUMENT", StatusCode::INVALID_ARGUMENT)
      .value("DEADLINE_EXCEEDED", StatusCode::DEADLINE_EXCEEDED)
      .value("RESOURCE_EXHAUSTED", StatusCode::RESOURCE_EXHAUSTED)
      .value("UNAUTHENTICATED", StatusCode::UNAUTHENTICATED)
      .value("UNAVAILABLE", StatusCode::UNAVAILABLE)
      .value("UNIMPLEMENTED", StatusCode::UNIMPLEMENTED)
      .export_values();

  py::class_<Status>(m, "Status")
      .def(py::init<StatusCode, const std::string&>(),
           py::arg("code"),
           py::arg("message"))
      .def_property_readonly("code", &Status::code)
      .def_property_readonly("message", &Status::message)
      .def_property_readonly("ok", &Status::ok);

  py::class_<LogProbData>(m, "LogProbData")
      .def(py::init())
      .def_readwrite("token", &LogProbData::token)
      .def_readwrite("token_id", &LogProbData::token_id)
      .def_readwrite("logprob", &LogProbData::logprob);

  py::class_<LogProb>(m, "LogProb")
      .def(py::init())
      .def_readwrite("token", &LogProbData::token)
      .def_readwrite("token_id", &LogProbData::token_id)
      .def_readwrite("logprob", &LogProbData::logprob)
      .def_readwrite("top_logprobs", &LogProb::top_logprobs);

  py::class_<SequenceOutput>(m, "SequenceOutput")
      .def(py::init())
      .def_readwrite("index", &SequenceOutput::index)
      .def_readwrite("text", &SequenceOutput::text)
      .def_readwrite("finish_reason", &SequenceOutput::finish_reason)
      .def_readwrite("logprobs", &SequenceOutput::logprobs);

  py::class_<RequestOutput>(m, "RequestOutput")
      .def(py::init())
      .def_readwrite("prompt", &RequestOutput::prompt)
      .def_readwrite("status", &RequestOutput::status)
      .def_readwrite("outputs", &RequestOutput::outputs)
      .def_readwrite("usage", &RequestOutput::usage)
      .def_readwrite("finished", &RequestOutput::finished);

  auto llm_handler =
      py::class_<LLMHandler>(m, "LLMHandler")
          .def(py::init<const LLMHandler::Options&>(), py::arg("options"))
          .def("schedule_async",
               &LLMHandler::schedule_async,
               py::call_guard<py::gil_scoped_release>())
          .def("schedule_chat_async",
               &LLMHandler::schedule_chat_async,
               py::call_guard<py::gil_scoped_release>())
          .def("schedule_batch_async",
               &LLMHandler::schedule_batch_async,
               py::call_guard<py::gil_scoped_release>())
          .def("schedule_chat_batch_async",
               &LLMHandler::schedule_chat_batch_async,
               py::call_guard<py::gil_scoped_release>())
          .def("start",
               &LLMHandler::start,
               py::call_guard<py::gil_scoped_release>())
          .def("stop",
               &LLMHandler::stop,
               py::call_guard<py::gil_scoped_release>())
          .def("run_until_complete",
               &LLMHandler::run_until_complete,
               py::call_guard<py::gil_scoped_release>())
          .def("apply_chat_template",
               &LLMHandler::apply_chat_template,
               py::call_guard<py::gil_scoped_release>())
          .def("encode",
               &LLMHandler::encode,
               py::call_guard<py::gil_scoped_release>())
          .def("decode",
               &LLMHandler::decode,
               py::call_guard<py::gil_scoped_release>())
          .def("reset",
               &LLMHandler::reset,
               py::call_guard<py::gil_scoped_release>());

  // LLMHandler::Options
  py::class_<LLMHandler::Options>(llm_handler, "Options")
      .def(py::init())
      .def_readwrite("model_path", &LLMHandler::Options::model_path_)
      .def_readwrite("devices", &LLMHandler::Options::devices_)
      .def_readwrite("draft_model_path",
                     &LLMHandler::Options::draft_model_path_)
      .def_readwrite("draft_devices", &LLMHandler::Options::draft_devices_)
      .def_readwrite("block_size", &LLMHandler::Options::block_size_)
      .def_readwrite("max_cache_size", &LLMHandler::Options::max_cache_size_)
      .def_readwrite("max_memory_utilization",
                     &LLMHandler::Options::max_memory_utilization_)
      .def_readwrite("enable_prefix_cache",
                     &LLMHandler::Options::enable_prefix_cache_)
      .def_readwrite("enable_cuda_graph",
                     &LLMHandler::Options::enable_cuda_graph_)
      .def_readwrite("cuda_graph_max_seq_len",
                     &LLMHandler::Options::cuda_graph_max_seq_len_)
      .def_readwrite("cuda_graph_batch_sizes",
                     &LLMHandler::Options::cuda_graph_batch_sizes_)
      .def_readwrite("draft_cuda_graph_batch_sizes",
                     &LLMHandler::Options::draft_cuda_graph_batch_sizes_)
      .def_readwrite("max_tokens_per_batch",
                     &LLMHandler::Options::max_tokens_per_batch_)
      .def_readwrite("max_seqs_per_batch",
                     &LLMHandler::Options::max_seqs_per_batch_)
      .def_readwrite("num_speculative_tokens",
                     &LLMHandler::Options::num_speculative_tokens_)
      .def_readwrite("num_handling_threads",
                     &LLMHandler::Options::num_handling_threads_);
}

}  // namespace llm::csrc