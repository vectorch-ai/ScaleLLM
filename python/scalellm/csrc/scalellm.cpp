#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llm.h"
#include "request/stopping_criteria.h"
#include "sampling/parameters.h"

namespace llm {
namespace py = pybind11;

PYBIND11_MODULE(PY_MODULE_NAME, m) {
  // class SamplingParameter
  py::class_<SamplingParameter, std::shared_ptr<SamplingParameter>>(
      m, "SamplingParameter")
      .def(py::init())
      .def_readwrite("frequency_penalty", &SamplingParameter::frequency_penalty)
      .def_readwrite("presence_penalty", &SamplingParameter::presence_penalty)
      .def_readwrite("repetition_penalty",
                     &SamplingParameter::repetition_penalty)
      .def_readwrite("temperature", &SamplingParameter::temperature)
      .def_readwrite("top_p", &SamplingParameter::top_p)
      .def_readwrite("top_k", &SamplingParameter::top_k);

  // class StoppingCriteria
  py::class_<StoppingCriteria, std::shared_ptr<StoppingCriteria>>(
      m, "StoppingCriteria")
      .def(py::init())
      .def_readwrite("max_tokens", &StoppingCriteria::max_tokens)
      .def_readwrite("eos_token_id", &StoppingCriteria::eos_token_id)
      .def_readwrite("ignore_eos_token", &StoppingCriteria::ignore_eos_token)
      .def_readwrite("stop_token_ids", &StoppingCriteria::stop_token_ids);

  // class LLM
  py::class_<LLM, std::shared_ptr<LLM>>(m, "LLM")
      .def(py::init<const std::string&,
                    const SamplingParameter&,
                    const StoppingCriteria&,
                    int64_t,
                    const std::string>())
      .def("generate", &LLM::generate);
}

}  // namespace llm