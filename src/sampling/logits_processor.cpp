#include "logits_processor.h"

#include <torch/torch.h>

#include <algorithm>
#include <memory>

namespace llm {
std::unique_ptr<LogitsProcessor> LogitsProcessor::create(
    const SamplingParameters& params,
    torch::ScalarType dtype,
    const torch::Device& device) {
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
            params.frequency_penalties,
            params.presence_penalties,
            dtype,
            device));
  }
  if (std::any_of(params.repetition_penalties.begin(),
                  params.repetition_penalties.end(),
                  [](float t) { return t != 1.0; })) {
    processors.push_back(std::make_unique<RepetitionPenaltyLogitsProcessor>(
        params.repetition_penalties, dtype, device));
  }
  if (std::any_of(params.temperatures.begin(),
                  params.temperatures.end(),
                  [](float t) { return t != 1.0; })) {
    processors.push_back(std::make_unique<TemperatureLogitsProcessor>(
        params.temperatures, dtype, device));
  }

  return std::make_unique<LogitsProcessorList>(std::move(processors));
}

}  // namespace llm
