#include "server/llm.h"

#include <absl/strings/str_split.h>

#include "request/sequence.h"

namespace llm {

LLM::LLM(const std::string& model_path,
         const llm::SamplingParameter& sp,
         const llm::StoppingCriteria& sc,
         int64_t max_seq_len,
         const std::string& device_str)
      : sampling_param_(sp), stopping_criteria_(sc),
        max_seq_len_(max_seq_len) {
  auto devices = parse_devices(device_str);
  engine_ = new llm::Engine(devices);
  CHECK(engine_->init(model_path));
  block_manager_ = engine_->block_manager();
  tokenizer_ = engine_->tokenizer();
}

LLM::LLM(const std::string& model_path) {
  // simple test for python
}

void LLM::generate(const std::vector<std::string>& batched_prompt) {
  std::vector<llm::Sequence*> sequences;
  sequences.reserve(batched_prompt.size());

  for (size_t i = 0; i < batched_prompt.size(); ++i) {
    // create sequences
    std::vector<int> prompt_tokens;
    tokenizer_->encode(batched_prompt[i], &prompt_tokens);

    auto sequence = new llm::Sequence(sampling_param_, stopping_criteria_,
        prompt_tokens, true, nullptr);
    sequences.emplace_back(sequence); 
  }

  for (int64_t i = 0; i < max_seq_len_; ++i) {
    sequences.erase(std::remove_if(sequences.begin(), sequences.end(),
                    [](llm::Sequence* seq) {
                      return seq->is_finished();
                    }), sequences.end());
    if (sequences.empty()) {
      break;
    }
    CHECK(block_manager_->allocate_slots_for_sequences(sequences));

    // run inference
    const auto output_parameters = engine_->execute_model(sequences);

    const auto& next_tokens = output_parameters.next_tokens;
    const int64_t num_seqs = next_tokens.numel();

    CHECK(num_seqs == sequences.size());

    const int64_t* new_token_ids = next_tokens.data_ptr<int64_t>();
    // process sequence in batch
    for (int64_t i = 0; i < num_seqs; ++i) {
      auto sequence = sequences[i];
      const int32_t next_token_id = static_cast<int32_t>(new_token_ids[i]);
      // add the next token to sequence and check if the sequence is finished
      if (!sequence->append_new_token_id(next_token_id)) {
        block_manager_->release_slots_for_sequence(sequence);
        continue;
      }
    }
  }
}

std::vector<torch::Device> LLM::parse_devices(
    const std::string& device_str) {
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
  CHECK(device_types.size() == 1)
      << "All devices must be of the same type.";
  return devices;
}

} // namespace llm
