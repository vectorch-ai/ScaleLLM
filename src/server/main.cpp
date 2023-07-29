#include <c10/core/TensorImpl.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>
#include <string>

#include "common/state_dict.h"
#include "models/llama/model_args.h"
#include "models/llama/transformer.h"
#include "tokenizer/sentencepiece_tokenizer.h"

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

  llm::SentencePieceTokenizer tokenizer(FLAGS_tokenizer_path);

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

  std::string prompt = "Enter some text: ";
  std::cout << prompt;

  // construct batch input [2, max_seq_len]
  const auto pad_id = tokenizer.pad_id();
  auto batch_tensor = torch::full({1, args.max_seq_len()}, pad_id);
  std::string input;
  while (std::getline(std::cin, input) && input != "exit") {
    if (input.empty()) {
      continue;
    }

    auto tokens = tokenizer.encode(input);
    int64_t prev_pos = 0;
    // generate tokens until the end of sentence token is generated
    for (int64_t cur_pos = tokens.size(); cur_pos < args.max_seq_len();
         ++cur_pos) {
      // [1, max_seq_len]
      batch_tensor.index_put_({0, torch::indexing::Slice(0, tokens.size())},
                              torch::tensor(tokens));

      // run inference
      const auto slice_tensor =
          batch_tensor.index({torch::indexing::Slice(),
                              torch::indexing::Slice(prev_pos, cur_pos)});

      const auto logits = transformer->forward(slice_tensor, prev_pos);

      const auto flatten_logits = logits.index({torch::indexing::Slice(), -1});
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

      // add the next token to the batch input
      batch_tensor.index_put_({torch::indexing::Slice(), cur_pos}, flat_tensor);
      prev_pos = cur_pos;

      const auto next_token_scalar =
          static_cast<int>(flat_tensor.item<int64_t>());
      // decode the output and print it
      std::cout << tokenizer.decode({next_token_scalar}) << " " << std::flush;

      if (next_token_scalar == tokenizer.eos_id()) {
        break;
      }
    }

    // print the prompt and wait for the next input
    std::cout << std::endl << prompt;
  }

  return 0;
}
