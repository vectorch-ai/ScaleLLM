#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "kernels/quantization/marlin.h"

namespace llm::csrc {
namespace py = pybind11;

void init_kernels(py::module_& m) {
  // marlin gemm kernels
  m.def("marlin_fp16_int4_gemm",
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

  m.def("marlin_gemm",
        &marlin::gptq_gemm,
        "Marlin GPTQ GEMM",
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("scales"),
        py::arg("zeros"),
        py::arg("g_idx"),
        py::arg("perm"),
        py::arg("workspace"),
        py::arg("num_bits"),
        py::arg("is_k_full"),
        py::arg("has_zp"),
        py::arg("use_fp32_reduce"));

  // marlink repack kernels
  m.def("marlin_gptq_repack",
        &marlin::gptq_repack,
        "Marlin GPTQ repack",
        py::arg("q_weight"),
        py::arg("perm"),
        py::arg("out"),
        py::arg("num_bits"));

  m.def("marlin_awq_repack",
        &marlin::awq_repack,
        "Marlin AWQ repack",
        py::arg("q_weight"),
        py::arg("out"),
        py::arg("num_bits"));
}

}  // namespace llm::csrc