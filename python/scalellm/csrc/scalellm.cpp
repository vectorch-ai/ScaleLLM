#include <folly/init/Init.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "_sampling_parameter.h"
#include "llm.h"

namespace llm {
namespace py = pybind11;

PYBIND11_MODULE(PY_MODULE_NAME, m) {
  // glog and glfag will be initialized in folly::init
  int argc = 0;
  char** argv = nullptr;
  folly::Init init(&argc, &argv);

  // class SamplingParameter
  py::class_<SamplingParameter_, std::shared_ptr<SamplingParameter_>>(
      m, "SamplingParameter")
      .def(py::init())
      .def_readwrite("frequency_penalty",
                     &SamplingParameter_::frequency_penalty)
      .def_readwrite("presence_penalty", &SamplingParameter_::presence_penalty)
      .def_readwrite("repetition_penalty",
                     &SamplingParameter_::repetition_penalty)
      .def_readwrite("temperature", &SamplingParameter_::temperature)
      .def_readwrite("top_p", &SamplingParameter_::top_p)
      .def_readwrite("top_k", &SamplingParameter_::top_k);

  // class LLM
  py::class_<LLM, std::shared_ptr<LLM>>(m, "LLM")
      .def(py::init<const std::string&,
                    const SamplingParameter_&,
                    int64_t,
                    const std::string>())
      .def("generate", &LLM::generate);
}

}  // namespace llm