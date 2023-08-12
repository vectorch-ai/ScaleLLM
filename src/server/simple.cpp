#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>
#include <string>

#include "hf_model_downloader.h"
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
  // initialize gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // test model download
  // const std::string model_path =
  // llm::download_hf_model("bigscience/bloom-1b1"); LOG(INFO) << "Model
  // downloaded to: " << model_path;

  torch::manual_seed(1);

  llm::SentencePieceTokenizer tokenizer(FLAGS_tokenizer_path);

  torch::InferenceMode guard;

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

  llm::Transformer transformer(args, 1);

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

  std::string prompt = "Enter a prompt: ";
  std::cout << prompt;

  std::string input;
  while (std::getline(std::cin, input) && input != "exit") {
    if (input.empty()) {
      continue;
    }

    std::string output;
    auto tokens = tokenizer.encode(input);
    int64_t prev_pos = 0;
    // generate tokens until the end of sentence token is generated
    for (auto cur_pos = static_cast<int64_t>(tokens.size());
         cur_pos < args.max_seq_len();
         ++cur_pos) {
      auto tokens_tensor = torch::tensor(tokens);
      // Creating positions based on the length of token_ids
      // Equivalent to Python's range(len(token_ids))
      std::vector<int64_t> positions(tokens.size());
      std::iota(positions.begin(), positions.end(), 0);
      torch::Tensor positions_tensor = torch::tensor(positions);
      std::vector<int64_t> cu_seq_lens = {0,
                                          static_cast<int64_t>(tokens.size())};

      // run inference
      const auto logits =
          transformer(tokens_tensor, positions_tensor, cu_seq_lens);

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
      tokens.push_back(next_token_scalar);

      // decode the output and print it
      const auto new_output = tokenizer.decode(tokens);
      std::cout << new_output.substr(output.size()) << std::flush;
      output = new_output;

      if (next_token_scalar == tokenizer.eos_id()) {
        break;
      }
    }

    // print the prompt and wait for the next input
    std::cout << std::endl << prompt;
  }

  return 0;
}
