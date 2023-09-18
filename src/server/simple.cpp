#include <absl/strings/str_split.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>
#include <string>

#include "engine/engine.h"
#include "engine/worker.h"
#include "hf_model_downloader.h"
#include "memory/block_manager.h"
#include "memory/kv_cache.h"
#include "model_loader/model_loader.h"
#include "models/model_args.h"
#include "models/parameters.h"
#include "request/sampling_parameter.h"
#include "request/sequence.h"
#include "request/stopping_criteria.h"
#include "model_loader/state_dict.h"

DEFINE_string(model_path,
              "/home/michael/code/llama/llama-2-7b",
              "Path to the model file.");
DEFINE_string(tokenizer_path,
              "/home/michael/code/llama/tokenizer.model",
              "Path to the tokenizer file.");

DEFINE_string(device, "cuda", "Device to run the model on.");

DEFINE_double(temperature, 0.6, "Temperature for sampling.");

DEFINE_double(top_p, 0.9, "Top p for sampling.");
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
  CHECK(!devices.empty()) << "No devices specified.";
  CHECK(device_types.size() == 1)
      << "All devices must be of the same type. Got: " << FLAGS_device;

  // set the default dtype
  torch::ScalarType dtype{};
  if (devices[0].is_cpu()) {
    // always use float32 on CPU since float16 is not supported
    dtype = torch::kFloat;
    LOG(INFO) << "Using float32 on CPU.";
  } else {
    dtype = torch::kHalf;
  }

  llm::Engine engine(dtype, devices);
  CHECK(engine.init(FLAGS_model_path, FLAGS_tokenizer_path));
  const auto* tokenizer = engine.tokenizer();
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
  stopping_criteria.eos_token_id = tokenizer->eos_id();

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
                           nullptr);

    // generate tokens until the end of sentence token is generated
    for (int64_t cur_pos = prompt_token_len; cur_pos < FLAGS_max_seq_len;
         ++cur_pos) {
      // allocate slots for the sequence
      CHECK(block_manager->allocate_slots_for_sequence(&sequence));

      // run inference
      const auto output_params = engine.execute_model({&sequence});

      torch::Tensor next_token = output_params.next_tokens;
      const auto flat_tensor = next_token.reshape({-1});

      // add the next token to the list of tokens
      const auto next_token_scalar = static_cast<int>(flat_tensor.item<int>());
      sequence.append_new_token_id(next_token_scalar);

      // decode the output and print delta
      std::cout << sequence.decode_delta_text(*tokenizer) << std::flush;

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
