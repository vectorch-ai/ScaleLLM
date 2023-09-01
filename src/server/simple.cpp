#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>
#include <string>

#include "engine/worker.h"
#include "hf_model_downloader.h"
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

torch::Tensor sample_top_p(torch::Tensor logits, float top_p) {
  auto [probs_sort, probs_idx] =
      torch::sort(logits, /*dim=*/-1, /*descending=*/true);
  // Calculate the cumulative sum of sorted probabilities
  const auto probs_sum = torch::cumsum(probs_sort, /*dim=*/-1);
  // Create a mask where (cumulative sum - current value) > p
  const auto mask = (probs_sum - probs_sort) > top_p;
  // Set values where mask is true to 0.0
  probs_sort.masked_fill_(mask, 0);
  // Normalize the probabilities
  probs_sort.div_(probs_sort.sum(-1, /*keepdim=*/true));
  // Sample from the adjusted distribution
  const auto selected = torch::multinomial(probs_sort, /*num_samples=*/1);
  // Get the original indices of the sampled values
  return torch::gather(probs_idx, /*dim=*/-1, selected);
}

int main(int argc, char* argv[]) {
  // initialize glog and gflags
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // test model download
  // const std::string model_path =
  // llm::download_hf_model("bigscience/bloom-1b1"); LOG(INFO) << "Model
  // downloaded to: " << model_path;

  llm::SentencePieceTokenizer tokenizer(FLAGS_tokenizer_path);

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

  llm::ModelLoader model_loader(FLAGS_model_path);

  llm::ModelArgs args = model_loader.model_args();
  if (args.vocab_size() == -1) {
    args.vocab_size(tokenizer.n_words());
  }
  // TODO: remove this two from model args
  args.max_seq_len(FLAGS_max_seq_len).max_batch_size(FLAGS_max_batch_size);

  llm::Worker worker(device);
  worker.init(args);

  // load the weights from the checkpoint
  for (const auto& state_dict : model_loader) {
    worker.load_state_dict(state_dict);
  }

  const int64_t block_size = 8;  // 8 slots per block
  const int64_t num_blocks = args.max_seq_len() / block_size + 1;

  // preallocate slots and blocks for the query
  auto block_tables = torch::arange(0, num_blocks, torch::kInt).unsqueeze(0);

  std::string prompt = "Enter a prompt: ";
  std::cout << prompt;

  std::string input;
  while (std::getline(std::cin, input) && input != "exit") {
    if (input.empty()) {
      continue;
    }

    std::string output;
    auto tokens_ids = tokenizer.encode(input);
    int64_t prev_pos = 0;
    // generate tokens until the end of sentence token is generated
    for (int64_t cur_pos = static_cast<int64_t>(tokens_ids.size());
         cur_pos < args.max_seq_len();
         ++cur_pos) {
      auto tokens = torch::tensor(tokens_ids, torch::kLong);
      auto slice_tokens =
          tokens.slice(/*dim=*/0, /*start=*/prev_pos, /*end=*/cur_pos);
      torch::Tensor positions = torch::arange(prev_pos, cur_pos, torch::kLong);
      llm::InputParameters input_params;
      // manually assign the slot ids and block tables
      input_params.block_tables = block_tables;
      input_params.slot_ids = torch::arange(prev_pos, cur_pos, torch::kInt);
      if (prev_pos == 0) {
        // prefill
        input_params.num_prompt_tokens = cur_pos;
        const std::vector<int32_t> cu_seq_lens = {
            0, static_cast<int32_t>(cur_pos)};
        input_params.cu_seq_lens = torch::tensor(cu_seq_lens, torch::kInt);
        input_params.context_lens = torch::tensor({0}, torch::kInt);
        input_params.max_seq_len = static_cast<int32_t>(cur_pos);
        input_params.sample_idx = torch::tensor({cur_pos - 1}, torch::kInt);
        input_params.token_ids =
            torch::tensor(tokens_ids, torch::kInt).unsqueeze(0);
      } else {
        // generate
        input_params.num_prompt_tokens = 0;
        const std::vector<int32_t> context_lens = {
            static_cast<int32_t>(cur_pos)};
        input_params.cu_seq_lens = torch::tensor({0}, torch::kInt);
        input_params.context_lens = torch::tensor(context_lens, torch::kInt);
        input_params.max_context_len = static_cast<int32_t>(cur_pos);
        input_params.sample_idx = torch::tensor({0}, torch::kInt);
        input_params.token_ids =
            torch::tensor(tokens_ids, torch::kInt).unsqueeze(0);
      }

      llm::SamplingParameters sampling_params;
      llm::SamplingParameter param;
      param.temperature = FLAGS_temperature;
      param.top_p = FLAGS_top_p;
      param.seed = 1;
      sampling_params.add(param);

      // run inference
      const auto output_params = worker.execute_model(
          slice_tokens, positions, input_params, sampling_params);

      torch::Tensor next_token = output_params.next_tokens;
      const auto flat_tensor = next_token.reshape({-1});

      // add the next token to the list of tokens
      const auto next_token_scalar = static_cast<int>(flat_tensor.item<int>());
      tokens_ids.push_back(next_token_scalar);

      // decode the output and print it
      const auto new_output = tokenizer.decode(tokens_ids);
      std::cout << new_output.substr(output.size()) << std::flush;
      output = new_output;

      prev_pos = cur_pos;
      if (next_token_scalar == tokenizer.eos_id()) {
        break;
      }
    }

    // print the prompt and wait for the next input
    std::cout << std::endl << prompt;
  }

  return 0;
}
