#include "server/llm.h"

#include <absl/strings/str_split.h>

#include "engine/llm_engine.h"
#include "request/sequence.h"

namespace llm {

LLM::LLM(const std::string& model_path,
         const llm::SamplingParameter& sp,
         const llm::StoppingCriteria& sc,
         int64_t max_seq_len,
         const std::string& device_str)
    : sampling_param_(sp), stopping_criteria_(sc), max_seq_len_(max_seq_len) {
  auto devices = parse_devices(device_str);
  LLMEngine::Options options;
  options.devices(devices);
  engine_ = new LLMEngine(options);
  CHECK(engine_->init(model_path));
  block_manager_ = engine_->block_manager();
  tokenizer_ = engine_->tokenizer();
}

void LLM::generate(const std::vector<std::string>& batched_prompt) {
  std::vector<llm::Sequence*> sequences;
  sequences.reserve(batched_prompt.size());

  for (size_t i = 0; i < batched_prompt.size(); ++i) {
    // create sequences
    std::vector<int> prompt_tokens;
    tokenizer_->encode(batched_prompt[i], &prompt_tokens);

    Sequence::Options options;
    options.sampling_param = sampling_param_;
    options.stopping_criteria = stopping_criteria_;
    options.echo = true;
    const size_t capacity = prompt_tokens.size() + max_seq_len_ + 1;
    auto* sequence =
        new Sequence(batched_prompt[i], prompt_tokens, capacity, options);
    sequences.emplace_back(sequence);
  }

  for (int64_t i = 0; i < max_seq_len_; ++i) {
    sequences.erase(
        std::remove_if(sequences.begin(),
                       sequences.end(),
                       [](llm::Sequence* seq) { return seq->is_finished(); }),
        sequences.end());
    if (sequences.empty()) {
      break;
    }
    CHECK(block_manager_->allocate_blocks_for(sequences));

    // run inference
    Batch batch(sequences);
    engine_->execute_model(batch);
    // process sequence in batch
    for (auto* sequence : sequences) {
      // add the next token to sequence and check if the sequence is finished
      if (sequence->is_finished()) {
        block_manager_->release_blocks_for(sequence);
      }
    }
  }
}

std::vector<torch::Device> LLM::parse_devices(const std::string& device_str) {
  std::vector<torch::Device> devices;
  if (device_str == "auto") {
    // use all available gpus if any
    const auto num_gpus = torch::cuda::device_count();
    if (num_gpus == 0) {
      LOG(INFO) << "no gpus found, using cpu.";
      return {torch::kCPU};
    }
    devices.reserve(num_gpus);
    for (int i = 0; i < num_gpus; ++i) {
      devices.emplace_back(torch::kCUDA, i);
    }
    return devices;
  }

  // parse device string
  const std::vector<std::string> device_strs = absl::StrSplit(device_str, ',');
  std::set<torch::DeviceType> device_types;
  devices.reserve(device_strs.size());
  for (const auto& device_str : device_strs) {
    devices.emplace_back(device_str);
    device_types.insert(devices.back().type());
  }
  CHECK(!devices.empty()) << "No devices specified.";
  CHECK(device_types.size() == 1) << "All devices must be of the same type.";
  return devices;
}

}  // namespace llm
