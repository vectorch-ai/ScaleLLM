#include <absl/strings/str_split.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/embed.h>

#include <filesystem>
#include <iostream>
#include <string>

#include "engine/engine.h"
#include "request/sequence.h"
#include "request/stopping_criteria.h"
#include "sampling/parameters.h"

DEFINE_string(model_name_or_path,
              "THUDM/chatglm3-6b",
              "hf model name or path to the model file.");

DEFINE_string(model_allow_patterns,
              "*.json,*.tiktoken,*.model,*.safetensors",
              "Allow patterns for model files.");

DEFINE_string(device,
              "cuda",
              "Device to run the model on, e.g. cpu, cuda:0, cuda:0,cuda:1, or "
              "auto to use all available gpus.");

DEFINE_int32(max_seq_len, 256, "Maximum sequence length.");

DEFINE_double(temperature, 0, "Temperature for sampling.");

DEFINE_double(top_p, 1.0, "Top p for sampling.");
DEFINE_int64(top_k, 0, "Top k for sampling.");

DEFINE_double(repetition_penalty, 1.0, "Repetition penalty for sampling.");

DEFINE_double(frequency_penalty, 0.0, "Frequency penalty for sampling.");
DEFINE_double(presence_penalty, 0.0, "Presence penalty for sampling.");

std::string download_model(const std::string& model_name) {
  namespace py = pybind11;
  py::scoped_interpreter guard{};  // Start the interpreter

  py::dict globals = py::globals();
  globals["repo_id"] = model_name;
  globals["allow_patterns"] = FLAGS_model_allow_patterns;
  py::exec(R"(
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(repo_id, allow_patterns=allow_patterns.split(','))
  )",
           globals,
           globals);
  return globals["model_path"].cast<std::string>();
}

std::vector<torch::Device> parse_devices(const std::string& device_str) {
  std::vector<torch::Device> devices;
  if (device_str == "auto") {
    // use all available gpus if any
    const auto num_gpus = torch::cuda::device_count();
    if (num_gpus == 0) {
      LOG(INFO) << "no gpus found, using cpu.";
      return {torch::kCPU};
    }
    devices.reserve(num_gpus);
    for (int i = 0; i < num_gpus; ++i) {
      devices.emplace_back(torch::kCUDA, i);
    }
    return devices;
  }

  // parse device string
  const std::vector<std::string> device_strs = absl::StrSplit(device_str, ',');
  std::set<torch::DeviceType> device_types;
  devices.reserve(device_strs.size());
  for (const auto& device_str : device_strs) {
    devices.emplace_back(device_str);
    device_types.insert(devices.back().type());
  }
  CHECK(!devices.empty()) << "No devices specified.";
  CHECK(device_types.size() == 1)
      << "All devices must be of the same type. Got: " << FLAGS_device;
  return devices;
}

std::string to_string(const std::vector<torch::Device>& devices) {
  std::stringstream ss;
  for (size_t i = 0; i < devices.size(); ++i) {
    const auto& device = devices[i];
    if (i == 0) {
      ss << device;
    } else {
      ss << "," << device;
    }
  }
  return ss.str();
}

int main(int argc, char* argv[]) {
  // initialize glog and gflags
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // check if model path exists
  std::string model_path = FLAGS_model_name_or_path;
  if (!std::filesystem::exists(model_path)) {
    // not a model path, try to download the model from huggingface hub
    model_path = download_model(FLAGS_model_name_or_path);
  }

  // parse devices
  const auto devices = parse_devices(FLAGS_device);
  LOG(INFO) << "Using devices: " << to_string(devices);

  llm::Engine engine(devices);
  CHECK(engine.init(model_path));
  auto tokenizer = engine.tokenizer();
  llm::BlockManager* block_manager = engine.block_manager();
  const auto& model_args = engine.model_args();

  llm::SamplingParameter sampling_param;
  sampling_param.temperature = FLAGS_temperature;
  sampling_param.top_p = FLAGS_top_p;
  sampling_param.top_k = FLAGS_top_k;
  sampling_param.repetition_penalty = FLAGS_repetition_penalty;
  sampling_param.frequency_penalty = FLAGS_frequency_penalty;
  sampling_param.presence_penalty = FLAGS_presence_penalty;

  llm::StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = FLAGS_max_seq_len;
  stopping_criteria.ignore_eos_token = false;
  stopping_criteria.eos_token_id = model_args.eos_token_id();
  stopping_criteria.stop_token_ids = model_args.stop_token_ids();

  std::string prompt = "Enter a prompt: ";
  std::cout << prompt;

  std::string input;
  while (std::getline(std::cin, input) && input != "exit") {
    if (input.empty()) {
      continue;
    }

    // create a request
    std::vector<int> prompt_tokens;
    tokenizer->encode(input, &prompt_tokens);
    const int64_t prompt_token_len = static_cast<int64_t>(prompt_tokens.size());

    llm::Sequence sequence(
        sampling_param, stopping_criteria, prompt_tokens, true, nullptr);

    // generate tokens until the end of sentence token is generated
    for (int64_t cur_pos = prompt_token_len; cur_pos < FLAGS_max_seq_len;
         ++cur_pos) {
      // allocate slots for the sequence
      CHECK(block_manager->allocate_blocks_for(&sequence));

      // run inference
      const auto output_params = engine.execute_model(&sequence);

      torch::Tensor next_token = output_params.next_tokens;
      const auto flat_tensor = next_token.view({-1});

      // add the next token to the list of tokens
      const auto next_token_id = static_cast<int>(flat_tensor.item<int>());
      if (!sequence.append_new_token_id(next_token_id)) {
        // sequence is finished
        break;
      }

      // decode the output and print delta
      std::cout << sequence.decode_delta_text(sequence.num_tokens(), *tokenizer)
                << std::flush;
    }

    // release the slots for the sequence
    block_manager->release_blocks_for(&sequence);

    // print the prompt and wait for the next input
    std::cout << "\n\n" << prompt;
  }

  return 0;
}
