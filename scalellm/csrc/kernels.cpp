#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "kernels/quantization/marlin/marlin.h"

namespace llm::csrc {
namespace py = pybind11;

// Add two tensors for snanity check
torch::Tensor add_test(const torch::Tensor& a, const torch::Tensor& b) {
  return a + b;
}

void init_kernels(py::module_& m) {
  m.def("add_test", &add_test, "Add two tensors");

  // marlin gemm kernels
  m.def("fp16_int4_gemm_marlin",
        &marlin::fp16_int4_gemm,
        "Marlin FP16xINT4 GEMM",
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("s"),
        py::arg("workspace"),
        py::arg("thread_k") = -1,
        py::arg("thread_n") = -1,
        py::arg("sms") = -1,
        py::arg("max_par") = 8);
}

}  // namespace llm::csrc