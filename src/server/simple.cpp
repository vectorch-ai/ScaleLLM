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
#include "models/input_parameters.h"
#include "models/llama/transformer.h"
#include "models/model_args.h"
#include "models/model_loader.h"
#include "tokenizer/sentencepiece_tokenizer.h"
#include "torch_utils/state_dict.h"

DEFINE_string(model_path,
              "/home/michael/code/llama/llama-2-7b",
              "Path to the model file.");
DEFINE_string(tokenizer_path,
              "/home/michael/code/llama/tokenizer.model",
              "Path to the tokenizer file.");

DEFINE_string(device, "cuda", "Device to run the model on.");

DEFINE_double(temperature, 0.6, "Temperature for sampling.");

DEFINE_double(top_p, 0.9, "Top p for sampling.");

DEFINE_int32(max_seq_len, 256, "Maximum sequence length.");

DEFINE_int32(max_batch_size, 4, "Maximum batch size.");

int main(int argc, char* argv[]) {
  // initialize glog and gflags
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  torch::InferenceMode guard;
  torch::Device device(FLAGS_device);

  // set the default dtype
  if (device.is_cpu()) {
    // always use float32 on CPU since float16 is not supported
    torch::set_default_dtype(
        torch::scalarTypeToTypeMeta(torch::ScalarType::Float));
    LOG(INFO) << "Using float32 on CPU.";
  } else {
    torch::set_default_dtype(
        torch::scalarTypeToTypeMeta(torch::ScalarType::BFloat16));
  }

  llm::Engine engine({device});
  engine.init(FLAGS_model_path, FLAGS_tokenizer_path);
  const auto args = engine.model_args();
  const auto* tokenizer = engine.tokenizer();

  const int64_t block_size = 8;  // 8 slots per block
  const int64_t num_blocks = FLAGS_max_seq_len / block_size + 1;
  llm::BlockManager block_manager(num_blocks, block_size);

  std::string prompt = "Enter a prompt: ";
  std::cout << prompt;

  std::string input;
  while (std::getline(std::cin, input) && input != "exit") {
    if (input.empty()) {
      continue;
    }

    // create a request
    auto tokens_ids = tokenizer->encode(input);
    int64_t prompt_token_len = tokens_ids.size();
    llm::Sequence sequence;
    sequence.prompt = input;
    sequence.token_ids = std::move(tokens_ids);
    llm::Request request;
    request.sequences.push_back(std::move(sequence));
    request.sampling_param.temperature = FLAGS_temperature;
    request.sampling_param.top_p = FLAGS_top_p;

    std::string output;
    int64_t prev_pos = 0;
    // generate tokens until the end of sentence token is generated
    for (int64_t cur_pos = prompt_token_len; cur_pos < FLAGS_max_seq_len;
         ++cur_pos) {
      // allocate slots for the request
      block_manager.allocate_slots_for_request(&request);

      // run inference
      const auto output_params = engine.execute_model({&request});

      torch::Tensor next_token = output_params.next_tokens;
      const auto flat_tensor = next_token.reshape({-1});

      // add the next token to the list of tokens
      const auto next_token_scalar = static_cast<int>(flat_tensor.item<int>());
      request.sequences[0].append_new_token_id(next_token_scalar);

      // decode the output and print it
      const auto new_output = tokenizer->decode(request.sequences[0].token_ids);
      std::cout << new_output.substr(output.size()) << std::flush;
      output = new_output;

      prev_pos = cur_pos;
    }

    // release the slots for the request
    block_manager.release_slots_for_request(&request);

    // print the prompt and wait for the next input
    std::cout << std::endl << prompt;
  }

  return 0;
}
