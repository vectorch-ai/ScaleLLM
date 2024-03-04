#include <pybind11/pybind11.h>

#include "server/llm.h"

PYBIND11_MODULE(gen_py_wrappers, m) {
  // Add class LLM
  pybind11::class_<llm::LLM, std::shared_ptr<llm::LLM>>(m, "LLM")
    .def(pybind11::init<const std::string&, const llm::SamplingParameter&,
                  const llm::StoppingCriteria&, int64_t,
                  const std::string>())
    .def(pybind11::init<const std::string>())
    .def("generate", &llm::LLM::generate);

  // Add function add
  // m.def("add", &add, "A function which adds two numbers");
}
