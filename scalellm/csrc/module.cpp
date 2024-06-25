#include <folly/init/Init.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/metrics.h"

namespace llm::csrc {
namespace py = pybind11;

extern void init_sampling_params(py::module_& m);
extern void init_output(py::module_& m);
extern void init_llm_handler(py::module_& m);
extern void init_vlm_handler(py::module_& m);

// NOLINTNEXTLINE
static std::string get_metrics() { return Metrics::Instance().GetString(); }

PYBIND11_MODULE(PY_MODULE_NAME, m) {
  m.doc() = "ScaleLLM C++ extension";
  // glog and glfag will be initialized in folly::init
  //   int argc = 0;
  //   char** argv = nullptr;
  //   folly::Init init(&argc, &argv);
  m.def("get_metrics", &get_metrics);

  init_sampling_params(m);
  init_output(m);
  init_llm_handler(m);
  init_vlm_handler(m);
}

}  // namespace llm::csrc
