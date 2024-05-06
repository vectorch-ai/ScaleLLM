#include <folly/init/Init.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llm.h"
#include "llm_handler.h"
#include "request/status.h"
#include "sampling_params.h"

namespace llm::csrc {
namespace py = pybind11;

PYBIND11_MODULE(PY_MODULE_NAME, m) {
  // glog and glfag will be initialized in folly::init
  //   int argc = 0;
  //   char** argv = nullptr;
  //   folly::Init init(&argc, &argv);

  // class SamplingParameter
  py::class_<SamplingParams>(m, "SamplingParams")
      .def(py::init())
      .def_readwrite("frequency_penalty", &SamplingParams::frequency_penalty)
      .def_readwrite("presence_penalty", &SamplingParams::presence_penalty)
      .def_readwrite("repetition_penalty", &SamplingParams::repetition_penalty)
      .def_readwrite("temperature", &SamplingParams::temperature)
      .def_readwrite("top_p", &SamplingParams::top_p)
      .def_readwrite("top_k", &SamplingParams::top_k);

  py::class_<ChatMessage>(m, "ChatMessage")
      .def(py::init())
      .def_readwrite("role", &ChatMessage::role)
      .def_readwrite("content", &ChatMessage::content);

  py::class_<Statistics>(m, "Statistics")
      .def(py::init())
      .def_readwrite("num_prompt_tokens", &Statistics::num_prompt_tokens)
      .def_readwrite("num_generated_tokens", &Statistics::num_generated_tokens)
      .def_readwrite("num_total_tokens", &Statistics::num_total_tokens);

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
      .def(py::init<StatusCode, const std::string&>())
      .def_property_readonly("code", &Status::code)
      .def_property_readonly("message", &Status::message)
      .def_property_readonly("ok", &Status::ok);

  py::class_<SequenceOutput>(m, "SequenceOutput")
      .def(py::init())
      .def_readwrite("index", &SequenceOutput::index)
      .def_readwrite("text", &SequenceOutput::text)
      .def_readwrite("finish_reason", &SequenceOutput::finish_reason);

  py::class_<RequestOutput>(m, "RequestOutput")
      .def(py::init())
      .def_readwrite("status", &RequestOutput::status)
      .def_readwrite("outputs", &RequestOutput::outputs)
      .def_readwrite("stats", &RequestOutput::stats)
      .def_readwrite("finished", &RequestOutput::finished);

  py::class_<LLMHandler>(m, "LLMHandler")
      .def(py::init<const std::string&, const std::string&>())
      .def("schedule",
           &LLMHandler::schedule,
           py::call_guard<py::gil_scoped_release>())
      .def("stop", &LLMHandler::stop, py::call_guard<py::gil_scoped_release>());

  // class LLM
  py::class_<LLM, std::shared_ptr<LLM>>(m, "LLM")
      .def(py::init<const std::string&,
                    const SamplingParams&,
                    int64_t,
                    const std::string>())
      .def("generate", &LLM::generate);
}

}  // namespace llm::csrc