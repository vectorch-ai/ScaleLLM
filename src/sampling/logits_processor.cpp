#include "logits_processor.h"

#include <torch/torch.h>

#include <memory>

namespace llm {
std::unique_ptr<LogitsProcessor> LogitsProcessor::create(
    const SamplingParameters& params) {
  std::vector<std::unique_ptr<LogitsProcessor>> processors;

  // construct logits processors based on the given parameters
  // always try to skip creating a processor if possible
  if (params.frequency_penalties.defined()) {
    processors.push_back(
        std::make_unique<FrequencyPresencePenaltyLogitsProcessor>(
            params.frequency_penalties, params.presence_penalties));
  }

  if (params.repetition_penalties.defined()) {
    processors.push_back(std::make_unique<RepetitionPenaltyLogitsProcessor>(
        params.repetition_penalties));
  }

  if (params.temperatures.defined()) {
    processors.push_back(
        std::make_unique<TemperatureLogitsProcessor>(params.temperatures));
  }

  if (params.top_k.defined() || params.top_p.defined()) {
    processors.push_back(
        std::make_unique<TopKTopPLogitsProcessor>(params.top_k, params.top_p));
  }

  return std::make_unique<LogitsProcessorList>(std::move(processors));
}

}  // namespace llm
