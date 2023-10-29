#include <absl/strings/str_split.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <gflags/gflags.h>

#include <filesystem>
#include <iostream>
#include <string>

#include "common/logging.h"
#include "engine/engine.h"
#include "model_loader/model_downloader.h"
#include "models/args.h"
#include "models/input_parameters.h"
#include "request/sampling_parameter.h"
#include "request/sequence.h"
#include "request/stopping_criteria.h"

DEFINE_string(model_name_or_path,
              "TheBloke/Llama-2-7B-GPTQ",
              "hf model name or path to the model file.");

DEFINE_string(device, "cuda:0", "Device to run the model on.");

DEFINE_int32(max_seq_len, 256, "Maximum sequence length.");

DEFINE_double(temperature, 0, "Temperature for sampling.");

DEFINE_double(top_p, 1.0, "Top p for sampling.");
DEFINE_int64(top_k, 0, "Top k for sampling.");

DEFINE_double(repetition_penalty, 1.0, "Repetition penalty for sampling.");

DEFINE_double(frequency_penalty, 0.0, "Frequency penalty for sampling.");
DEFINE_double(presence_penalty, 0.0, "Presence penalty for sampling.");

int main(int argc, char* argv[]) {
  // initialize glog and gflags
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  torch::InferenceMode guard;

  // split device into chunks
  const std::vector<std::string> device_strs =
      absl::StrSplit(FLAGS_device, ',');
  std::vector<torch::Device> devices;
  devices.reserve(device_strs.size());
  std::set<torch::DeviceType> device_types;
  for (const auto& device_str : device_strs) {
    devices.emplace_back(device_str);
    device_types.insert(devices.back().type());
  }
  GCHECK(!devices.empty()) << "No devices specified.";
  GCHECK(device_types.size() == 1)
      << "All devices must be of the same type. Got: " << FLAGS_device;

  // set the default dtype
  torch::ScalarType dtype{};
  if (devices[0].is_cpu()) {
    // always use float32 on CPU since float16 is not supported
    dtype = torch::kFloat;
    GLOG(INFO) << "Using float32 on CPU.";
  } else {
    dtype = torch::kHalf;
  }

  // check if model path exists
  std::string model_path = FLAGS_model_name_or_path;
  if (!std::filesystem::exists(model_path)) {
    // not a model path, try to download the model from huggingface hub
    model_path = llm::hf::download_model(FLAGS_model_name_or_path);
  }

  llm::Engine engine(dtype, devices);
  GCHECK(engine.init(model_path));
  auto tokenizer = engine.tokenizer();
  llm::BlockManager* block_manager = engine.block_manager();

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
  // stopping_criteria.eos_token_id = tokenizer->eos_id();

  std::string prompt = "Enter a prompt: ";
  std::cout << prompt;

  std::string input;
  while (std::getline(std::cin, input) && input != "exit") {
    if (input.empty()) {
      continue;
    }

    // create a request
    std::vector<int> tokens_ids;
    tokenizer->encode(input, &tokens_ids);
    int64_t prompt_token_len = tokens_ids.size();

    llm::Sequence sequence(std::move(input),
                           std::move(tokens_ids),
                           &sampling_param,
                           &stopping_criteria,
                           nullptr,
                           /*echo=*/true);

    // generate tokens until the end of sentence token is generated
    for (int64_t cur_pos = prompt_token_len; cur_pos < FLAGS_max_seq_len;
         ++cur_pos) {
      // allocate slots for the sequence
      GCHECK(block_manager->allocate_slots_for_sequence(&sequence));

      // run inference
      const auto output_params = engine.execute_model({&sequence});

      torch::Tensor next_token = output_params.next_tokens;
      const auto flat_tensor = next_token.view({-1});

      // add the next token to the list of tokens
      const auto next_token_scalar = static_cast<int>(flat_tensor.item<int>());
      sequence.append_new_token_id(next_token_scalar);

      // decode the output and print delta
      std::cout << sequence.decode_delta_text(sequence.num_tokens(), *tokenizer)
                << std::flush;

      if (sequence.check_stopping_creteria()) {
        break;
      }
    }

    // release the slots for the sequence
    block_manager->release_slots_for_sequence(&sequence);

    // print the prompt and wait for the next input
    std::cout << std::endl << prompt;
  }

  return 0;
}
