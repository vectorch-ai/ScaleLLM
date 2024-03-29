#include "logits_processor.h"

#include <torch/torch.h>

#include <algorithm>
#include <memory>

namespace llm {
std::unique_ptr<LogitsProcessor> LogitsProcessor::create(
    const SamplingParameters& params,
    const torch::TensorOptions& options) {
  std::vector<std::unique_ptr<LogitsProcessor>> processors;

  // construct logits processors based on the given parameters
  // always try to skip creating a processor if possible
  if (std::any_of(params.frequency_penalties.begin(),
                  params.frequency_penalties.end(),
                  [](float t) { return t != 0.0; }) ||
      std::any_of(params.presence_penalties.begin(),
                  params.presence_penalties.end(),
                  [](float t) { return t != 0.0; })) {
    processors.push_back(
        std::make_unique<FrequencyPresencePenaltyLogitsProcessor>(
            params.frequency_penalties, params.presence_penalties, options));
  }
  if (std::any_of(params.repetition_penalties.begin(),
                  params.repetition_penalties.end(),
                  [](float t) { return t != 1.0; })) {
    processors.push_back(std::make_unique<RepetitionPenaltyLogitsProcessor>(
        params.repetition_penalties, options));
  }
  if (std::any_of(params.temperatures.begin(),
                  params.temperatures.end(),
                  [](float t) { return t != 0.0 && t != 1.0; })) {
    processors.push_back(std::make_unique<TemperatureLogitsProcessor>(
        params.temperatures, options));
  }

  const bool has_top_k = std::any_of(params.top_k.begin(),
                                     params.top_k.end(),
                                     [](int64_t t) { return t != 0; });
  const bool has_top_p = std::any_of(params.top_p.begin(),
                                     params.top_p.end(),
                                     [](float t) { return t != 1.0; });
  if (has_top_k || has_top_p) {
    processors.push_back(std::make_unique<TopKTopPLogitsProcessor>(
        params.top_k, params.top_p, options));
  }

  return std::make_unique<LogitsProcessorList>(std::move(processors));
}

}  // namespace llm
