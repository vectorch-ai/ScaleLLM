#include "handlers/sampling_params.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace llm::csrc {
namespace py = pybind11;
using namespace pybind11::literals;

void init_sampling_params(py::module_& m) {
  // class SamplingParameter
  py::class_<SamplingParams>(m, "SamplingParams")
      .def(py::init<uint32_t,                /*max_tokens*/
                    uint32_t,                /*n*/
                    std::optional<uint32_t>, /*best_of*/
                    bool,                    /*echo*/
                    float,                   /*frequency_penalty*/
                    float,                   /*presence_penalty*/
                    float,                   /*repetition_penalty*/
                    float,                   /*temperature*/
                    float,                   /*top_p*/
                    int64_t,                 /*top_k*/
                    bool,                    /*logprobs*/
                    int64_t,                 /*top_logprobs*/
                    bool,                    /*skip_special_tokens*/
                    bool,                    /*ignore_eos*/
                    std::optional<std::vector<std::string>>, /*stop*/
                    std::optional<std::vector<int32_t>>>(),  /*stop_token_ids*/
           py::arg("max_tokens") = 16,
           py::arg("n") = 1,
           py::arg("best_of") = std::nullopt,
           py::arg("echo") = false,
           py::arg("frequency_penalty") = 0.0,
           py::arg("presence_penalty") = 0.0,
           py::arg("repetition_penalty") = 1.0,
           py::arg("temperature") = 1.0,
           py::arg("top_p") = 1.0,
           py::arg("top_k") = -1,
           py::arg("logprobs") = false,
           py::arg("top_logprobs") = 0,
           py::arg("skip_special_tokens") = true,
           py::arg("ignore_eos") = false,
           py::arg("stop") = std::nullopt,
           py::arg("stop_token_ids") = std::nullopt)
      .def_readwrite("max_tokens", &SamplingParams::max_tokens)
      .def_readwrite("n", &SamplingParams::n)
      .def_readwrite("best_of", &SamplingParams::best_of)
      .def_readwrite("echo", &SamplingParams::echo)
      .def_readwrite("frequency_penalty", &SamplingParams::frequency_penalty)
      .def_readwrite("presence_penalty", &SamplingParams::presence_penalty)
      .def_readwrite("repetition_penalty", &SamplingParams::repetition_penalty)
      .def_readwrite("temperature", &SamplingParams::temperature)
      .def_readwrite("top_p", &SamplingParams::top_p)
      .def_readwrite("top_k", &SamplingParams::top_k)
      .def_readwrite("logprobs", &SamplingParams::logprobs)
      .def_readwrite("top_logprobs", &SamplingParams::top_logprobs)
      .def_readwrite("skip_special_tokens",
                     &SamplingParams::skip_special_tokens)
      .def_readwrite("ignore_eos", &SamplingParams::ignore_eos)
      .def_readwrite("stop", &SamplingParams::stop)
      .def_readwrite("stop_token_ids", &SamplingParams::stop_token_ids)
      .def("__repr__", [](const SamplingParams& self) {
        return "SamplingParams(max_tokens={}, n={}, best_of={}, echo={}, "
               "frequency_penalty={}, presence_penalty={}, "
               "repetition_penalty={}, temperature={}, top_p={}, top_k={}, "
               "logprobs={}, top_logprobs={}, skip_special_tokens={}, "
               "ignore_eos={}, stop={}, stop_token_ids={})"_s.format(
                   self.max_tokens,
                   self.n,
                   self.best_of,
                   self.echo,
                   self.frequency_penalty,
                   self.presence_penalty,
                   self.repetition_penalty,
                   self.temperature,
                   self.top_p,
                   self.top_k,
                   self.logprobs,
                   self.top_logprobs,
                   self.skip_special_tokens,
                   self.ignore_eos,
                   self.stop,
                   self.stop_token_ids);
      });
}

}  // namespace llm::csrc