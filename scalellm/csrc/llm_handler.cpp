#include "handlers/llm_handler.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace llm::csrc {
namespace py = pybind11;
using namespace pybind11::literals;

void init_llm_handler(py::module_& m) {
  py::class_<Message>(m, "Message")
      .def(py::init<const std::string&, const std::string&>(),
           py::arg("role"),
           py::arg("content"))
      .def_readwrite("role", &Message::role)
      .def_readwrite("content", &Message::content)
      .def("__repr__", [](const Message& self) {
        return "Message({}: {!r})"_s.format(self.role, self.content);
      });

  py::enum_<Priority>(m, "Priority")
      .value("DEFAULT", Priority::NORMAL)
      .value("LOW", Priority::LOW)
      .value("NORMAL", Priority::NORMAL)
      .value("HIGH", Priority::HIGH)
      .export_values();

  py::class_<std::future<bool>>(m, "Future")
      .def("wait",
           &std::future<bool>::wait,
           py::call_guard<py::gil_scoped_release>())
      .def("get",
           &std::future<bool>::get,
           py::call_guard<py::gil_scoped_release>());

  py::class_<BatchFuture>(m, "BatchFuture")
      .def("wait", &BatchFuture::wait, py::call_guard<py::gil_scoped_release>())
      .def("get", &BatchFuture::get, py::call_guard<py::gil_scoped_release>());

  auto llm_handler =
      py::class_<LLMHandler>(m, "LLMHandler")
          .def(py::init<const LLMHandler::Options&>(), py::arg("options"))
          .def("schedule_async",
               &LLMHandler::schedule_async,
               py::call_guard<py::gil_scoped_release>())
          .def("schedule_chat_async",
               &LLMHandler::schedule_chat_async,
               py::call_guard<py::gil_scoped_release>())
          .def("schedule_batch_async",
               &LLMHandler::schedule_batch_async,
               py::call_guard<py::gil_scoped_release>())
          .def("schedule_chat_batch_async",
               &LLMHandler::schedule_chat_batch_async,
               py::call_guard<py::gil_scoped_release>())
          .def("start",
               &LLMHandler::start,
               py::call_guard<py::gil_scoped_release>())
          .def("stop",
               &LLMHandler::stop,
               py::call_guard<py::gil_scoped_release>())
          .def("run_until_complete",
               &LLMHandler::run_until_complete,
               py::call_guard<py::gil_scoped_release>())
          .def("apply_chat_template",
               &LLMHandler::apply_chat_template,
               py::call_guard<py::gil_scoped_release>())
          .def("encode",
               &LLMHandler::encode,
               py::call_guard<py::gil_scoped_release>())
          .def("decode",
               &LLMHandler::decode,
               py::call_guard<py::gil_scoped_release>())
          .def("reset",
               &LLMHandler::reset,
               py::call_guard<py::gil_scoped_release>())
          .def("__repr__", [](const LLMHandler& self) {
            return "LLMHandler({})"_s.format(self.options());
          });

  // LLMHandler::Options
  py::class_<LLMHandler::Options>(llm_handler, "Options")
      .def(py::init())
      .def_readwrite("model_path", &LLMHandler::Options::model_path_)
      .def_readwrite("devices", &LLMHandler::Options::devices_)
      .def_readwrite("draft_model_path",
                     &LLMHandler::Options::draft_model_path_)
      .def_readwrite("draft_devices", &LLMHandler::Options::draft_devices_)
      .def_readwrite("block_size", &LLMHandler::Options::block_size_)
      .def_readwrite("max_cache_size", &LLMHandler::Options::max_cache_size_)
      .def_readwrite("max_memory_utilization",
                     &LLMHandler::Options::max_memory_utilization_)
      .def_readwrite("enable_prefix_cache",
                     &LLMHandler::Options::enable_prefix_cache_)
      .def_readwrite("enable_cuda_graph",
                     &LLMHandler::Options::enable_cuda_graph_)
      .def_readwrite("cuda_graph_max_seq_len",
                     &LLMHandler::Options::cuda_graph_max_seq_len_)
      .def_readwrite("cuda_graph_batch_sizes",
                     &LLMHandler::Options::cuda_graph_batch_sizes_)
      .def_readwrite("draft_cuda_graph_batch_sizes",
                     &LLMHandler::Options::draft_cuda_graph_batch_sizes_)
      .def_readwrite("max_tokens_per_batch",
                     &LLMHandler::Options::max_tokens_per_batch_)
      .def_readwrite("max_seqs_per_batch",
                     &LLMHandler::Options::max_seqs_per_batch_)
      .def_readwrite("num_speculative_tokens",
                     &LLMHandler::Options::num_speculative_tokens_)
      .def_readwrite("num_handling_threads",
                     &LLMHandler::Options::num_handling_threads_)
      .def("__repr__", [](const LLMHandler::Options& self) {
        return "Options(model_path={}, devices={}, draft_model_path={}, "
               "draft_devices={}, block_size={}, max_cache_size={}, "
               "max_memory_utilization={}, enable_prefix_cache={}, "
               "enable_cuda_graph={}, cuda_graph_max_seq_len={}, "
               "cuda_graph_batch_sizes={}, draft_cuda_graph_batch_sizes={}, "
               "max_tokens_per_batch={}, max_seqs_per_batch={}, "
               "num_speculative_tokens={}, num_handling_threads={})"_s.format(
                   self.model_path_,
                   self.devices_,
                   self.draft_model_path_,
                   self.draft_devices_,
                   self.block_size_,
                   self.max_cache_size_,
                   self.max_memory_utilization_,
                   self.enable_prefix_cache_,
                   self.enable_cuda_graph_,
                   self.cuda_graph_max_seq_len_,
                   self.cuda_graph_batch_sizes_,
                   self.draft_cuda_graph_batch_sizes_,
                   self.max_tokens_per_batch_,
                   self.max_seqs_per_batch_,
                   self.num_speculative_tokens_,
                   self.num_handling_threads_);
      });
}

}  // namespace llm::csrc