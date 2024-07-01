#include "handlers/vlm_handler.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace llm::csrc {
namespace py = pybind11;
using namespace pybind11::literals;

void init_vlm_handler(py::module_& m) {
  auto vlm_handler =
      py::class_<VLMHandler>(m, "VLMHandler")
          .def(py::init<const VLMHandler::Options&>(), py::arg("options"))
          .def("schedule_async",
               &VLMHandler::schedule_async,
               py::call_guard<py::gil_scoped_release>())
          .def("start",
               &VLMHandler::start,
               py::call_guard<py::gil_scoped_release>())
          .def("stop",
               &VLMHandler::stop,
               py::call_guard<py::gil_scoped_release>())
          .def("run_until_complete",
               &VLMHandler::run_until_complete,
               py::call_guard<py::gil_scoped_release>())
          .def("encode",
               &VLMHandler::encode,
               py::call_guard<py::gil_scoped_release>())
          .def("decode",
               &VLMHandler::decode,
               py::call_guard<py::gil_scoped_release>())
          .def("reset",
               &VLMHandler::reset,
               py::call_guard<py::gil_scoped_release>())
          .def("__repr__", [](const VLMHandler& self) {
            return "VLMHandler({})"_s.format(self.options());
          });

  // VLMHandler::Options
  py::class_<VLMHandler::Options>(vlm_handler, "Options")
      .def(py::init())
      .def_readwrite("model_path", &VLMHandler::Options::model_path_)
      .def_readwrite("devices", &VLMHandler::Options::devices_)
      .def_readwrite("block_size", &VLMHandler::Options::block_size_)
      .def_readwrite("max_cache_size", &VLMHandler::Options::max_cache_size_)
      .def_readwrite("max_memory_utilization",
                     &VLMHandler::Options::max_memory_utilization_)
      .def_readwrite("enable_prefix_cache",
                     &VLMHandler::Options::enable_prefix_cache_)
      .def_readwrite("enable_cuda_graph",
                     &VLMHandler::Options::enable_cuda_graph_)
      .def_readwrite("cuda_graph_max_seq_len",
                     &VLMHandler::Options::cuda_graph_max_seq_len_)
      .def_readwrite("cuda_graph_batch_sizes",
                     &VLMHandler::Options::cuda_graph_batch_sizes_)
      .def_readwrite("max_tokens_per_batch",
                     &VLMHandler::Options::max_tokens_per_batch_)
      .def_readwrite("max_seqs_per_batch",
                     &VLMHandler::Options::max_seqs_per_batch_)
      .def_readwrite("num_handling_threads",
                     &VLMHandler::Options::num_handling_threads_)
      .def_readwrite("image_input_type",
                     &VLMHandler::Options::image_input_type_)
      .def_readwrite("image_token_id", &VLMHandler::Options::image_token_id_)
      .def_readwrite("image_input_shape",
                     &VLMHandler::Options::image_input_shape_)
      .def_readwrite("image_feature_size",
                     &VLMHandler::Options::image_feature_size_)
      .def("__repr__", [](const VLMHandler::Options& self) {
        return "Options(model_path={}, devices={}, "
               "block_size={}, max_cache_size={}, "
               "max_memory_utilization={}, enable_prefix_cache={}, "
               "enable_cuda_graph={}, cuda_graph_max_seq_len={}, "
               "cuda_graph_batch_sizes={}, "
               "max_tokens_per_batch={}, max_seqs_per_batch={}, "
               "num_handling_threads={}, "
               "image_input_type={}, image_token_id={}, "
               "image_input_shape={}, image_feature_size={})"_s.format(
                   self.model_path_,
                   self.devices_,
                   self.block_size_,
                   self.max_cache_size_,
                   self.max_memory_utilization_,
                   self.enable_prefix_cache_,
                   self.enable_cuda_graph_,
                   self.cuda_graph_max_seq_len_,
                   self.cuda_graph_batch_sizes_,
                   self.max_tokens_per_batch_,
                   self.max_seqs_per_batch_,
                   self.num_handling_threads_,
                   self.image_input_type_,
                   self.image_token_id_,
                   self.image_input_shape_,
                   self.image_feature_size_);
      });
}

}  // namespace llm::csrc
