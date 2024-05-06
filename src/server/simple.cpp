#include <absl/strings/str_split.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/embed.h>

#include <filesystem>
#include <iostream>
#include <string>

#include "engine/engine_factory.h"
#include "request/sequence.h"
#include "request/stopping_criteria.h"
#include "sampling/parameters.h"
using namespace llm;
namespace py = pybind11;

DEFINE_string(model_name_or_path,
              "THUDM/chatglm3-6b",
              "hf model name or path to the model file.");

DEFINE_string(draft_model_name_or_path,
              "",
              "hf model name or path to the model file.");

DEFINE_string(model_allow_patterns,
              "*.json,*.tiktoken,*.model,*.safetensors",
              "Allow patterns for model files.");

DEFINE_string(device,
              "cuda",
              "Device to run the model on, e.g. cpu, cuda:0, cuda:0,cuda:1, or "
              "auto to use all available gpus.");

DEFINE_string(
    draft_device,
    "cuda",
    "Device to run the draft model on, e.g. cpu, cuda:0, cuda:0,cuda:1, or "
    "auto to use all available gpus.");

DEFINE_int32(max_seq_len, 256, "Maximum sequence length.");

DEFINE_double(temperature, 0, "Temperature for sampling.");

DEFINE_double(top_p, 1.0, "Top p for sampling.");
DEFINE_int64(top_k, 0, "Top k for sampling.");

DEFINE_double(repetition_penalty, 1.0, "Repetition penalty for sampling.");

DEFINE_double(frequency_penalty, 0.0, "Frequency penalty for sampling.");
DEFINE_double(presence_penalty, 0.0, "Presence penalty for sampling.");

DECLARE_int32(num_speculative_tokens);

std::string download_model(const std::string& model_name) {
  py::dict locals;
  locals["repo_id"] = model_name;
  locals["allow_patterns"] = FLAGS_model_allow_patterns;
  py::exec(R"(
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(repo_id, allow_patterns=allow_patterns.split(','))
  )",
           py::globals(),
           locals);
  return locals["model_path"].cast<std::string>();
}

int main(int argc, char* argv[]) {
  // initialize glog and gflags
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // check if model path exists
  std::string model_path = FLAGS_model_name_or_path;
  std::string draft_model_path = FLAGS_draft_model_name_or_path;
  {
    py::scoped_interpreter guard{};  // Start the interpreter
    CHECK(!model_path.empty()) << "model_name_or_path is empty.";
    if (!std::filesystem::exists(model_path)) {
      // not a model path, try to download the model from huggingface hub
      model_path = download_model(model_path);
    }
    if (!draft_model_path.empty() &&
        !std::filesystem::exists(draft_model_path)) {
      // not a model path, try to download the model from huggingface hub
      draft_model_path = download_model(draft_model_path);
    }
  }

  std::unique_ptr<Engine> engine = EngineFactory::create(
      model_path, FLAGS_device, draft_model_path, FLAGS_draft_device);

  auto tokenizer = engine->tokenizer();
  BlockManager* block_manager = engine->block_manager();
  const auto& model_args = engine->model_args();

  SamplingParameter sampling_param;
  sampling_param.temperature = FLAGS_temperature;
  sampling_param.top_p = FLAGS_top_p;
  sampling_param.top_k = FLAGS_top_k;
  sampling_param.repetition_penalty = FLAGS_repetition_penalty;
  sampling_param.frequency_penalty = FLAGS_frequency_penalty;
  sampling_param.presence_penalty = FLAGS_presence_penalty;

  StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = FLAGS_max_seq_len;
  stopping_criteria.ignore_eos_token = false;
  stopping_criteria.eos_token_id = model_args.eos_token_id();
  stopping_criteria.stop_token_ids = model_args.stop_token_ids();
  stopping_criteria.max_context_len =
      model_args.max_position_embeddings() - FLAGS_num_speculative_tokens;

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
    const size_t capacity = prompt_tokens.size() + FLAGS_max_seq_len +
                            FLAGS_num_speculative_tokens + 1;

    Sequence::Options options;
    options.sampling_param = sampling_param;
    options.stopping_criteria = stopping_criteria;
    options.echo = true;

    Sequence sequence(input, prompt_tokens, capacity, options);

    // allocate all slots for the sequence
    CHECK(block_manager->allocate_blocks_for(&sequence, capacity));

    // generate tokens until the end of sentence token is generated
    while (sequence.num_generated_tokens() < FLAGS_max_seq_len) {
      // run inference
      Batch batch(&sequence);
      engine->execute_model(batch);

      // check if sequence is finished
      if (sequence.is_finished()) {
        break;
      }

      // decode the output and print delta
      std::cout << sequence.decode_delta_text(sequence.token_ids(), *tokenizer)
                << std::flush;
    }

    // release the slots for the sequence
    block_manager->release_blocks_for(&sequence);

    // print the prompt and wait for the next input
    std::cout << "\n\n" << prompt;
  }

  return 0;
}
