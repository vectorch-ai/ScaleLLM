#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/state_dict.h"
#include "models/llama/model_args.h"
#include "models/llama/transformer.h"
#include "tokenizer/sentencepiece_tokenizer.h"

DEFINE_string(model_path,
              "/home/michael/code/llama/llama-2-7b-chat/consolidated.00.pth",
              "Path to the model file.");
DEFINE_string(tokenizer_path,
              "/home/michael/code/llama/tokenizer.model",
              "Path to the tokenizer file.");

int main(int argc, char* argv[]) {
  // initialize gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  llm::SentencePieceTokenizer tokenizer(FLAGS_tokenizer_path);

  llm::ModelArgs args;
  args.max_seq_len(512).max_batch_size(4);
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
  const auto weights =
      llm::StateDict::load_from_file(FLAGS_model_path, torch::kCPU);
  transformer->load_state_dict(weights);
  LOG(ERROR) << "Successfully loaded model. " << transformer;
  // move the model to the GPU
  // transformer->to(torch::kCUDA);
  // set the module in evaluation/inference mode
  transformer->eval();

  // construct a dummy input and run inference
  auto input = torch::randint(0, 100, {1, 512});
  LOG(ERROR) << "Input: " << input.sizes();
  auto output = transformer->forward(input, 0);
  LOG(ERROR) << "Output: " << output.sizes();

  return 0;
}
