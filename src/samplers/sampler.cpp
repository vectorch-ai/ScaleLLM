#include "sampler.h"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <functional>
namespace llm {
namespace {
torch::Generator make_generator(const torch::Device& device) {
  torch::Generator generator;
  if (device.type() == at::kCPU) {
    generator = torch::make_generator<torch::CPUGeneratorImpl>();
  } else if (device.type() == at::kCUDA) {
    generator = torch::make_generator<torch::CUDAGeneratorImpl>(device.index());
  } else {
    AT_ERROR("Device type ",
             c10::DeviceTypeName(device.type()),
             " is not supported for torch.Generator() api.");
  }
  return generator;
}

}  // namespace

Sampler::Sampler(const std::vector<bool>& do_sample,
                 const std::vector<uint64_t>& seeds,
                 const torch::Device& device) {
  CHECK_EQ(do_sample.size(), seeds.size());
  sample_funcs_.reserve(do_sample.size());
  for (bool sample : do_sample) {
    if (sample) {
      torch::optional<torch::Generator> generator;
      // use global generator when seed is 0
      // TODO: we should set seed for each request instead of for each token
      // if (seeds[i] != 0) {
      //   // create a generator for each sequence based on device type
      //   generator = make_generator(device);
      //   generator->set_current_seed(seeds[i]);
      // }

      sample_funcs_.emplace_back(
          [generator = std::move(generator)](const torch::Tensor& logits) {
            const auto probs = logits.softmax(/*dim=*/-1);
            return torch::multinomial(probs,
                                      /*num_samples=*/1,
                                      /*replacement=*/false,
                                      /*generator=*/generator);
          });
    } else {
      sample_funcs_.emplace_back([](const torch::Tensor& logits) {
        return logits.argmax(/*dim=*/-1);
      });
    }
  }
}

torch::Tensor Sampler::sample(const torch::Tensor& logits) const {
  const auto num_seqs = logits.size(0);
  CHECK_EQ(num_seqs, static_cast<int64_t>(sample_funcs_.size()));

  auto output = torch::empty(
      {num_seqs}, torch::TensorOptions(torch::kInt64).device(logits.device()));
  // sample logits for each sequence
  for (int64_t i = 0; i < num_seqs; ++i) {
    auto sample = sample_funcs_[i](logits[i]);
    output.index_put_({i}, sample);
  }
  return output;
}

}  // namespace llm
