#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace llm::csrc {
namespace py = pybind11;

// Add two tensors for snanity check
torch::Tensor add_test(const torch::Tensor& a, const torch::Tensor& b) {
  return a + b;
}

void init_kernels(py::module_& m) {
  m.def("add_test", &add_test, "Add two tensors");
}

}  // namespace llm::csrc