#include "request/output.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "request/status.h"

namespace llm::csrc {
namespace py = pybind11;
using namespace pybind11::literals;

void init_output(py::module_& m) {
  py::class_<Usage>(m, "Usage")
      .def(py::init())
      .def_readwrite("num_prompt_tokens", &Usage::num_prompt_tokens)
      .def_readwrite("num_generated_tokens", &Usage::num_generated_tokens)
      .def_readwrite("num_total_tokens", &Usage::num_total_tokens)
      .def("__repr__", [](const Usage& self) {
        return "Usage(num_prompt_tokens={}, num_generated_tokens={}, num_total_tokens={})"_s
            .format(self.num_prompt_tokens,
                    self.num_generated_tokens,
                    self.num_total_tokens);
      });

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
      .def_property_readonly("ok", &Status::ok)
      .def("__repr__", [](const Status& self) {
        if (self.message().empty()) {
          return "Status(code={})"_s.format(self.code());
        }
        return "Status(code={}, message={!r})"_s.format(self.code(),
                                                        self.message());
      });

  py::class_<LogProbData>(m, "LogProbData")
      .def(py::init())
      .def_readwrite("token", &LogProbData::token)
      .def_readwrite("token_id", &LogProbData::token_id)
      .def_readwrite("logprob", &LogProbData::logprob)
      .def_readwrite("finished_token", &LogProbData::finished_token)
      .def("__repr__", [](const LogProbData& self) {
        return "LogProb(token={!r}, logprob={})"_s.format(self.token,
                                                          self.logprob);
      });

  py::class_<LogProb>(m, "LogProb")
      .def(py::init())
      .def_readwrite("token", &LogProbData::token)
      .def_readwrite("token_id", &LogProbData::token_id)
      .def_readwrite("logprob", &LogProbData::logprob)
      .def_readwrite("finished_token", &LogProbData::finished_token)
      .def_readwrite("top_logprobs", &LogProb::top_logprobs)
      .def("__repr__", [](const LogProb& self) {
        return "LogProb(token={!r}, logprob={}, top_logprobs={})"_s.format(
            self.token, self.logprob, self.top_logprobs);
      });

  py::class_<SequenceOutput>(m, "SequenceOutput")
      .def(py::init())
      .def_readwrite("index", &SequenceOutput::index)
      .def_readwrite("text", &SequenceOutput::text)
      .def_readwrite("token_ids", &SequenceOutput::token_ids)
      .def_readwrite("finish_reason", &SequenceOutput::finish_reason)
      .def_readwrite("logprobs", &SequenceOutput::logprobs)
      .def("__repr__", [](const SequenceOutput& self) {
        return "SequenceOutput({}: {!r})"_s.format(self.index, self.text);
      });

  py::class_<RequestOutput>(m, "RequestOutput")
      .def(py::init())
      .def_readwrite("prompt", &RequestOutput::prompt)
      .def_readwrite("status", &RequestOutput::status)
      .def_readwrite("outputs", &RequestOutput::outputs)
      .def_readwrite("usage", &RequestOutput::usage)
      .def_readwrite("finished", &RequestOutput::finished)
      .def("__repr__", [](const RequestOutput& self) {
        return "RequestOutput({}, {}, {})"_s.format(
            self.outputs, self.status, self.usage);
      });
}

}  // namespace llm::csrc