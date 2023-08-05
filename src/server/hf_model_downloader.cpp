#include "hf_model_downloader.h"

#include <pybind11/embed.h>

#include <string>

namespace llm {
std::string download_hf_model(const std::string& model_name) {
  namespace py = pybind11;
  py::scoped_interpreter guard{};  // Start the interpreter

  py::module_ hub = py::module_::import("huggingface_hub");
  std::string cache_dir = "/data";
  return hub
      .attr("snapshot_download")(model_name, py::arg("cache_dir") = cache_dir)
      .cast<std::string>();
}

}  // namespace llm
