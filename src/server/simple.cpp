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
#include "models/llama/transformer.h"
#include "models/model_args.h"
#include "models/model_loader.h"
#include "models/parameters.h"
#include "request/sampling_parameter.h"
#include "request/sequence.h"
#include "request/stopping_criteria.h"
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

  llm::SamplingParameter sampling_param;
  sampling_param.temperature = FLAGS_temperature;
  sampling_param.top_p = FLAGS_top_p;

  llm::StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = FLAGS_max_seq_len;
  stopping_criteria.ignore_eos_token = true;

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

    llm::Sequence sequence(std::move(input),
                           std::move(tokens_ids),
                           &sampling_param,
                           &stopping_criteria);

    // generate tokens until the end of sentence token is generated
    for (int64_t cur_pos = prompt_token_len; ;
         ++cur_pos) {
      // allocate slots for the sequence
      CHECK(block_manager.allocate_slots_for_sequence(&sequence));

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
    block_manager.release_slots_for_sequence(&sequence);

    // print the prompt and wait for the next input
    std::cout << std::endl << prompt;
  }

  return 0;
}
