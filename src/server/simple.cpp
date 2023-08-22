#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>
#include <string>

#include "hf_model_downloader.h"
#include "memory/kv_cache.h"
#include "models/input_parameters.h"
#include "models/llama/transformer.h"
#include "models/model_args.h"
#include "tokenizer/sentencepiece_tokenizer.h"
#include "torch_utils/state_dict.h"

DEFINE_string(model_path,
              "/home/michael/code/llama/llama-2-7b/consolidated.00.pth",
              "Path to the model file.");
DEFINE_string(tokenizer_path,
              "/home/michael/code/llama/tokenizer.model",
              "Path to the tokenizer file.");

DEFINE_string(device, "cpu", "Device to run the model on.");

DEFINE_double(temperature, 0.6, "Temperature for sampling.");

DEFINE_double(top_p, 0.9, "Top p for sampling.");

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

  llm::ModelArgs args;
  args.max_seq_len(128).max_batch_size(4);
  // TODO: read from params.json
  // {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32,
  // "norm_eps": 1e-06, "vocab_size": -1}
  args.dim(4096)
      .multiple_of(256)
      .n_heads(32)
      .n_layers(32)
      .norm_eps(1e-6)
      .vocab_size(-1);
  args.vocab_size(tokenizer.n_words());

  llm::Transformer transformer(args, 1, device);

  // load the weights from the checkpoint
  {
    const auto weights =
        llm::StateDict::load_from_file(FLAGS_model_path, torch::kCPU);
    transformer->load_state_dict(weights);
    LOG(INFO) << "Successfully loaded model. " << transformer;
  }
  // move the model to the GPU
  // transformer->to(torch::kCUDA);
  // set the module in evaluation/inference mode
  transformer->eval();

  // calculate cache shapes
  const auto element_size = sizeof(float);
  const auto x = 16 / element_size;
  const int64_t block_size = 4;  // 4 slots per block
  const int64_t num_heads = args.n_heads();
  const int64_t head_dim = args.dim() / num_heads;
  const int64_t num_blocks = args.max_seq_len() / block_size + 1;
  const std::vector<int64_t> key_cache_shape = {
      num_blocks, num_heads, static_cast<int64_t>(head_dim / x), block_size, x};
  const std::vector<int64_t> value_cache_shape = {
      num_blocks, num_heads, head_dim, block_size};

  // create a KVCache for each layer
  std::vector<llm::KVCache> kv_caches;
  kv_caches.reserve(args.n_layers());
  for (int i = 0; i < args.n_layers(); ++i) {
    auto key_cache = torch::zeros(key_cache_shape, device);
    auto value_cache = torch::zeros(value_cache_shape, device);
    kv_caches.emplace_back(key_cache, value_cache);
  }

  // preallocate slots and blocks for the query
  auto block_tables =
      torch::arange(0,
                    num_blocks,
                    torch::TensorOptions().dtype(torch::kInt).device(device))
          .unsqueeze(0);

  std::string prompt = "Enter a prompt: ";
  std::cout << prompt;

  std::string input;
  while (std::getline(std::cin, input) && input != "exit") {
    if (input.empty()) {
      continue;
    }
    // reset the random seed for each input
    torch::manual_seed(1);

    std::string output;
    auto tokens_ids = tokenizer.encode(input);
    int64_t prev_pos = 0;
    // generate tokens until the end of sentence token is generated
    for (int64_t cur_pos = static_cast<int64_t>(tokens_ids.size());
         cur_pos < args.max_seq_len();
         ++cur_pos) {
      auto tokens = torch::tensor(
          tokens_ids,
          torch::TensorOptions().dtype(torch::kLong).device(device));
      auto slice_tokens =
          tokens.slice(/*dim=*/0, /*start=*/prev_pos, /*end=*/cur_pos);
      torch::Tensor positions = torch::arange(
          prev_pos,
          cur_pos,
          torch::TensorOptions().dtype(torch::kLong).device(device));
      llm::InputParameters input_params;
      // manually assign the slot ids and block tables
      input_params.block_tables = block_tables;
      input_params.slot_ids = torch::arange(
          prev_pos,
          cur_pos,
          torch::TensorOptions().dtype(torch::kInt).device(device));
      if (prev_pos == 0) {
        // prefill
        input_params.cu_seq_lens = {0, cur_pos};
        input_params.context_lens =
            torch::tensor({},
                          torch::TensorOptions()
                              .dtype(torch::kInt)
                              .device(device));  // empty tensor
      } else {
        // generate
        input_params.cu_seq_lens = {0};
        input_params.context_lens = torch::tensor(
            {cur_pos},
            torch::TensorOptions().dtype(torch::kInt).device(device));
      }

      // run inference
      const auto logits =
          transformer(slice_tokens, positions, kv_caches, input_params);

      const auto flatten_logits = logits.index({-1});
      // sample the next token
      torch::Tensor next_token;
      if (FLAGS_temperature > 0.0) {
        const auto logits_scaled = flatten_logits / FLAGS_temperature;
        const auto probs = torch::softmax(logits_scaled, /*dim=*/-1);
        next_token = sample_top_p(probs, FLAGS_top_p);
      } else {
        next_token = torch::argmax(flatten_logits, /*dim=*/-1);
      }

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
