#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llm.h"
#include "request/stopping_criteria.h"
#include "sampling/parameters.h"

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  // class SamplingParameter
  py::class_<llm::SamplingParameter, std::shared_ptr<llm::SamplingParameter>>(
      m, "SamplingParameter")
      .def(py::init())
      .def_readwrite("frequency_penalty",
                     &llm::SamplingParameter::frequency_penalty)
      .def_readwrite("presence_penalty",
                     &llm::SamplingParameter::presence_penalty)
      .def_readwrite("repetition_penalty",
                     &llm::SamplingParameter::repetition_penalty)
      .def_readwrite("temperature", &llm::SamplingParameter::temperature)
      .def_readwrite("top_p", &llm::SamplingParameter::top_p)
      .def_readwrite("top_k", &llm::SamplingParameter::top_k);

  // class StoppingCriteria
  py::class_<llm::StoppingCriteria, std::shared_ptr<llm::StoppingCriteria>>(
      m, "StoppingCriteria")
      .def(py::init())
      .def_readwrite("max_tokens", &llm::StoppingCriteria::max_tokens)
      .def_readwrite("eos_token_id", &llm::StoppingCriteria::eos_token_id)
      .def_readwrite("ignore_eos_token",
                     &llm::StoppingCriteria::ignore_eos_token)
      .def_readwrite("stop_token_ids", &llm::StoppingCriteria::stop_token_ids);

  // class LLM
  py::class_<llm::LLM, std::shared_ptr<llm::LLM>>(m, "LLM")
      .def(py::init<const std::string&,
                    const llm::SamplingParameter&,
                    const llm::StoppingCriteria&,
                    int64_t,
                    const std::string>())
      .def("generate", &llm::LLM::generate);

  // function add
  // m.def("add", &add, "A function which adds two numbers");
}
