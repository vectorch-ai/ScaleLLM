#include "worker.h"

#include <c10/cuda/CUDAGuard.h>
#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <string>
#include <utility>

#include "executor.h"
#include "torch_utils/state_dict.h"
#include "models/input_parameters.h"

namespace llm {

bool Worker::load_state_dict(const StateDict& state_dict) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  executor_.schedule(
      [this, &state_dict, promise = std::move(promise)]() mutable {
        // load the model from the given path within the working thread
        // const bool success = this->load_state_dict(state_dict);
        promise.setValue(true);
      });
  // wait for the model to be loaded
  future.wait();
  if (!future.value()) {
    LOG(ERROR) << "Failed to load model from " << model_path_;
  }
  return future.value();
}

folly::SemiFuture<bool> Worker::execute_model_async(
    const torch::Tensor& tokens,     // [num_tokens]
    const torch::Tensor& positions,  // [num_tokens]
    const InputParameters& parameters) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  executor_.schedule([this,
                      tokens,
                      positions,
                      parameters,
                      promise = std::move(promise)]() mutable {
    // run the model on the given input in working thread
    this->execute_model(tokens, positions, parameters);
    promise.setValue(true);
  });
  return future;
}

// initialize model, cache manager. async call
folly::SemiFuture<bool> Worker::init_async() {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  executor_.schedule([this, promise = std::move(promise)]() mutable {
    // initialize model and cache manager within the working thread
    const bool success = this->init();
    promise.setValue(success);
  });
  return future;
}

void Worker::execute_model(const torch::Tensor& tokens,     // [num_tokens]
                           const torch::Tensor& positions,  // [num_tokens]
                           const InputParameters& parameters) const {
  // all tensors should be on the same device as model
  auto d_tokens = tokens.to(device_);
  auto d_positions = positions.to(device_);

  // call model forward and return the result
}

bool Worker::init() {
  if (device_.is_cuda()) {
    // set device for the working thread. the cuda runtime API is thread safe.
    // it maintains a thread local state for each thread.
    // just need to set once for each working thread.
    at::cuda::set_device(device_.index());
  }

  return true;
}

}  // namespace llm
