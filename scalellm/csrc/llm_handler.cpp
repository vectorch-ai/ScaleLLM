#include "handlers/llm_handler.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace llm::csrc {
namespace py = pybind11;

void init_llm_handler(py::module_& m) {
  py::class_<Message>(m, "Message")
      .def(py::init<const std::string&, const std::string&>(),
           py::arg("role"),
           py::arg("content"))
      .def_readwrite("role", &Message::role)
      .def_readwrite("content", &Message::content);

  py::enum_<Priority>(m, "Priority")
      .value("DEFAULT", Priority::NORMAL)
      .value("LOW", Priority::LOW)
      .value("NORMAL", Priority::NORMAL)
      .value("HIGH", Priority::HIGH)
      .export_values();

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
               py::call_guard<py::gil_scoped_release>());

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
                     &LLMHandler::Options::num_handling_threads_);
}

}  // namespace llm::csrc