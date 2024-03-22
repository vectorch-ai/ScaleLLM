#include <absl/strings/str_split.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <pybind11/embed.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "common/time.h"
#include "engine/engine.h"
#include "request/sequence.h"
#include "request/stopping_criteria.h"
#include "sampling/parameters.h"

DEFINE_string(model_name_or_path,
              "/data/llama2-7b",
              "hf model name or path to the model file.");

DEFINE_string(input_file, "/data/dataset/Chatbot_group_10_2.json", "");

DEFINE_string(model_allow_patterns, "*", "Allow patterns for model files.");

DEFINE_string(device,
              "cuda",
              "Device to run the model on, e.g. cpu, cuda:0, cuda:0,cuda:1, or "
              "auto to use all available gpus.");

DEFINE_int32(batch_size, 1, "Batch size.");
DEFINE_int32(max_seq_len, 100, "Maximum sequence length.");

DEFINE_double(temperature, 1, "Temperature for sampling.");

DEFINE_double(top_p, 0.01, "Top p for sampling.");
DEFINE_int64(top_k, 0, "Top k for sampling.");

DEFINE_double(repetition_penalty, 1.0, "Repetition penalty for sampling.");

DEFINE_double(frequency_penalty, 0.0, "Frequency penalty for sampling.");
DEFINE_double(presence_penalty, 0.0, "Presence penalty for sampling.");

DECLARE_int32(block_size);

class JsonFileReader {
 public:
  static std::vector<std::string> read_json_input(
      const std::string& input_file) {
    std::ifstream file(input_file);
    if (file.fail()) {
      LOG(ERROR) << "Unable to open the json file:" << input_file;
    }
    nlohmann::json data = nlohmann::json::parse(file);

    std::vector<std::string> result;
    result.reserve(data.size());
    for (auto& it : data) {
      result.emplace_back(it["prompt"]);
    }
    return result;
  }
};

static constexpr double kMicrosToSeconds = 1000000.0;

class LLM {
 public:
  explicit LLM(const std::string model_path,
               const llm::SamplingParameter& sp,
               const llm::StoppingCriteria& sc,
               const std::string device_str)
      : sampling_param_(sp), stopping_criteria_(sc) {
    auto devices = parse_devices(device_str);
    engine_ = new llm::Engine(devices);

    CHECK(engine_->init(FLAGS_model_name_or_path));
    block_manager_ = engine_->block_manager();
    tokenizer_ = engine_->tokenizer();

    const auto& args = engine_->model_args();
    stopping_criteria_.eos_token_id = args.eos_token_id();
    stopping_criteria_.stop_token_ids = args.stop_token_ids();
  }

  void generate(const std::vector<std::string>& batched_prompt) {
    std::vector<llm::Sequence*> sequences;
    sequences.reserve(batched_prompt.size());

    for (size_t i = 0; i < batched_prompt.size(); ++i) {
      // create sequences
      std::vector<int> prompt_tokens;
      tokenizer_->encode(batched_prompt[i], &prompt_tokens);

      auto sequence = new llm::Sequence(batched_prompt[i],
                                        prompt_tokens,
                                        sampling_param_,
                                        stopping_criteria_,
                                        /*echo=*/false,
                                        /*on_stream=*/nullptr);
      sequences.emplace_back(sequence);
    }

    llm::Time time;
    double request_cost = 0;

    for (int64_t i = 0; i < FLAGS_max_seq_len; ++i) {
      sequences.erase(
          std::remove_if(sequences.begin(),
                         sequences.end(),
                         [](llm::Sequence* seq) { return seq->is_finished(); }),
          sequences.end());
      if (sequences.empty()) {
        break;
      }
      CHECK(block_manager_->allocate_blocks_for(sequences));

      uint64_t time_start = time.now_micros();
      // run inference
      const auto output_parameters = engine_->execute_model(sequences);
      uint64_t time_end = time.now_micros();
      request_cost += time_end - time_start;

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
          block_manager_->release_blocks_for(sequence);
          continue;
        }
        // std::cout << sequence->decode_delta_text(sequence->num_tokens(),
        // *tokenizer_)
        //           << std::flush;
      }
    }

    // release all remaining slots
    block_manager_->release_blocks_for(sequences);

    std::cout << "request cost:" << request_cost / kMicrosToSeconds
              << std::endl;
  }

 private:
  std::vector<torch::Device> parse_devices(const std::string& device_str) {
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
    const std::vector<std::string> device_strs =
        absl::StrSplit(device_str, ',');
    std::set<torch::DeviceType> device_types;
    devices.reserve(device_strs.size());
    for (const auto& device_str : device_strs) {
      devices.emplace_back(device_str);
      device_types.insert(devices.back().type());
    }
    CHECK(!devices.empty()) << "No devices specified.";
    CHECK(device_types.size() == 1)
        << "All devices must be of the same type. Got: " << FLAGS_device;
    return devices;
  }

 private:
  llm::Engine* engine_;
  llm::BlockManager* block_manager_;
  llm::SamplingParameter sampling_param_;
  llm::StoppingCriteria stopping_criteria_;
  std::unique_ptr<llm::Tokenizer> tokenizer_;
};

int main(int argc, char* argv[]) {
  // initialize glog and gflags
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_minloglevel = google::INFO;

  llm::SamplingParameter sampling_param;
  sampling_param.temperature = 0;
  // sampling_param.temperature = FLAGS_temperature;
  // sampling_param.top_p = FLAGS_top_p;
  // sampling_param.top_k = FLAGS_top_k;
  // sampling_param.repetition_penalty = FLAGS_repetition_penalty;
  // sampling_param.frequency_penalty = FLAGS_frequency_penalty;
  // sampling_param.presence_penalty = FLAGS_presence_penalty;

  llm::StoppingCriteria stopping_criteria;
  stopping_criteria.max_tokens = FLAGS_max_seq_len;
  stopping_criteria.ignore_eos_token = false;
  // stopping_criteria.eos_token_id = tokenizer->eos_id();

  LLM llm_engine(FLAGS_model_name_or_path,
                 sampling_param,
                 stopping_criteria,
                 FLAGS_device);

  auto input_prompts = JsonFileReader::read_json_input(FLAGS_input_file);
  auto loop_count = ceil(input_prompts.size() / FLAGS_batch_size);

  llm::Time time;
  double total_cost = 0.0;

  for (int i = 0; i < loop_count; ++i) {
    auto begin = input_prompts.begin() + i * FLAGS_batch_size;
    auto end = input_prompts.begin() + (i + 1) * FLAGS_batch_size;
    std::vector<std::string> batched_input_prompt(begin, end);

    uint64_t time_start = time.now_micros();
    llm_engine.generate(batched_input_prompt);
    uint64_t time_end = time.now_micros();

    double cost = (time_end - time_start) / kMicrosToSeconds;
    std::cout << "one_batch_cost:" << cost << std::endl;
    total_cost += cost;
    std::cout << "average cost:" << total_cost / (i + 1) << std::endl;
  }

  std::cout << "average cost:" << total_cost / loop_count << std::endl;
  return 0;
}
