#include "scheduler/scheduler.h"
#include "scheduler/speculative_scheduler.h"

#include <absl/strings/str_split.h>
#include <c10/core/Device.h>
#include <memory>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace llm {

namespace {
const char* MODEL_DIR = "/data/llama2-7b";
}

class FakeSSMEngine : public Engine {
 public:
  FakeSSMEngine(const std::vector<torch::Device>& devices)
    : Engine(devices) {}
  virtual ~FakeSSMEngine() {}

  OutputParameters execute_model(const std::vector<Sequence*>&) override {
    if (spec_tokens_idx_ >= spec_token_ids_.size()) {
      GLOG(ERROR) << "Out of Range, you should setup FakeSSMEngine correctly.";
      return OutputParameters();
    }
    ++execute_model_calls_;
    OutputParameters output;
    std::vector<int64_t> val;
    val.emplace_back(spec_token_ids_[spec_tokens_idx_++]);
    
    output.next_tokens = torch::unsqueeze(
        torch::tensor(val, torch::kInt64), 0);

    return output;
  }

  OutputParameters validate(const std::vector<Sequence*>&) override {
    ++validate_calls_;
    return OutputParameters();
  }

  void set_spec_token_ids(const std::vector<int64_t>& spec_token_ids) {
    spec_token_ids_ = spec_token_ids;
  }

  int get_execute_model_calls() {
    return execute_model_calls_;
  }

  int get_validate_calls() {
    return validate_calls_;
  }

 private:
  std::vector<int64_t> spec_token_ids_;
  int spec_tokens_idx_ = 0;
  int execute_model_calls_ = 0;
  int validate_calls_ = 0;
};

class FakeLLMEngine : public Engine {
 public:
  FakeLLMEngine(const std::vector<torch::Device>& devices)
    : Engine(devices) {}
  virtual ~FakeLLMEngine() {}

  OutputParameters execute_model(const std::vector<Sequence*>&) override {
    ++execute_model_calls_;
    return OutputParameters();
  }

  OutputParameters validate(const std::vector<Sequence*>&) override {
    if (valid_tokens_idx_ >= valid_token_ids_.size()) {
      GLOG(ERROR) << "Out of Range, you should setup FakeLLMEngine correctly.";
      return OutputParameters();
    }

    ++validate_calls_;
    OutputParameters output;
    output.next_tokens = torch::unsqueeze(
        torch::tensor(valid_token_ids_, torch::kInt64), 0);
    return output;
  }

  void set_valid_token_id(const std::vector<int64_t>& valid_token_ids) {
    valid_token_ids_ = valid_token_ids;
  }

  int get_execute_model_calls() {
    return execute_model_calls_;
  }

  int get_validate_calls() {
    return validate_calls_;
  }

 private:
  std::vector<int64_t> valid_token_ids_;
  int valid_tokens_idx_ = 0;
  int execute_model_calls_ = 0;
  int validate_calls_ = 0;
};

class TestableSpeculativeScheduler {
 public:
  TestableSpeculativeScheduler(const std::vector<int64_t>& spec_token_ids,
      const std::vector<int64_t>& valid_token_id, uint64_t spec_steps) {
    create_scheduler_config(spec_steps);
    create_engine(spec_token_ids, valid_token_id);
    create_scheduler();
  }

  bool schedule(std::unique_ptr<Request>& request) {
    return scheduler_->schedule(request);
  }

  void step(const absl::Duration& timeout) {
    scheduler_->step(timeout);
  }

  Tokenizer* tokenizer() {
    return tokenizer_.get();
  }

  int get_ssm_execute_model_calls() {
    return ssm_engine_->get_execute_model_calls();
  }

  int get_llm_execute_model_calls() {
    return llm_engine_->get_execute_model_calls();
  }

  int get_ssm_validate_calls() {
    return ssm_engine_->get_validate_calls();;
  }

  int get_llm_validate_calls() {
    return llm_engine_->get_validate_calls();;
  }

 private:
  void create_engine(const std::vector<int64_t>& spec_token_ids,
      const std::vector<int64_t>& valid_token_id) {
    std::vector<torch::Device> devices;
    devices.emplace_back(torch::kCUDA, 0);
    llm_engine_ = std::make_unique<FakeLLMEngine>(devices);
    llm_engine_->init(MODEL_DIR);
    llm_engine_->set_valid_token_id(valid_token_id);
    tokenizer_ = llm_engine_->tokenizer();
    ssm_engine_ = std::make_unique<FakeSSMEngine>(devices);
    ssm_engine_->init(MODEL_DIR);
    ssm_engine_->set_spec_token_ids(spec_token_ids);
  }

  void create_scheduler_config(uint64_t spec_steps) {
    config_ = std::make_unique<SchedulerConfig>(
        SchedulerType::SPECULATIVE, SchedulerPolicyType::FCFS, spec_steps);
  }

  void create_scheduler() {
    scheduler_ = std::make_unique<SpeculativeScheduler>(
        *config_.get(), llm_engine_.get(), ssm_engine_.get());
  }

 private:
  std::unique_ptr<SpeculativeScheduler> scheduler_;
  std::unique_ptr<FakeLLMEngine> llm_engine_;
  std::unique_ptr<FakeSSMEngine> ssm_engine_;
  std::unique_ptr<SchedulerConfig> config_;
  std::unique_ptr<Tokenizer> tokenizer_;
};

class RequestWrapper {
 public:
  RequestWrapper(Tokenizer* tokenizer)
      : tokenizer_(tokenizer) {
    sampling_param_.temperature = 0;
    sampling_param_.top_p = 1.0;
    sampling_param_.top_k = 0;
    sampling_param_.repetition_penalty = 1.0;
    sampling_param_.frequency_penalty = 0.0;
    sampling_param_.presence_penalty = 0.0;

    stopping_criteria_.max_tokens = 100;
    stopping_criteria_.ignore_eos_token = false;
    stopping_criteria_.eos_token_id = 2;
  }

  std::unique_ptr<Request> create_request(
      const std::vector<int> prompt_tokens) {
    return create_request_impl(prompt_tokens);
  }

  std::unique_ptr<Request> create_request(const std::string& prompt) {
    std::vector<int> prompt_tokens;
    tokenizer_->encode(prompt, &prompt_tokens);
    return create_request_impl(prompt_tokens);
  }

  SequenceResult sequence_output() {
    return outputs_["req_001"];
  }

 private:
  std::unique_ptr<Request> create_request_impl(
      const std::vector<int> prompt_tokens) {
    const std::string request_id("req_001");
    auto request = std::make_unique<Request>(request_id, prompt_tokens);
    request->echo = true;
    request->sampling_param = sampling_param_;
    request->stopping_criteria = stopping_criteria_;
    request->on_finish = [this, &request_id] (
        const std::vector<SequenceResult>& seq_results,
        const Status& status, const Statistics& stats) -> bool {
      this->outputs_.emplace(request_id, seq_results[0]);
      return true;
    };
    request->add_sequence([this, request_id] (
        const std::string& delta, FinishReason reason) -> bool {
      SequenceResult output;
      output.output_text = delta;
      output.finish_reason = reason;
      this->outputs_.emplace(request_id, output);
      return true;
    });
    return request;
  }

 private:
  SamplingParameter sampling_param_;
  StoppingCriteria stopping_criteria_;
  Tokenizer* tokenizer_;

  std::map<std::string, SequenceResult> outputs_;
  bool is_finished = false;
};

TEST(SpeculativeSchedulerTest, Speculative5StepsPartiallyMatchTest) {
  // who[1058] is[338] mess[4473] i[29875] ?[29973]
  const std::vector<int64_t> spec_token_ids = {1058, 338, 4473, 29875, 29973};
  const std::vector<int64_t> valid_token_ids = {1058, 338, 4473, 1058, 338, 4473};
  const uint64_t spec_steps = 5;

  TestableSpeculativeScheduler scheduler(spec_token_ids, valid_token_ids,
      spec_steps);
  RequestWrapper request_wrapper(scheduler.tokenizer());

  std::vector<int> input_tokens = {1058};
  // expect output tokens: [1058] + [1058, 338, 4473, 1058] 
  // expect output string: who who is mess who
  const std::string expect_output("who who is mess who");
  auto request = request_wrapper.create_request(input_tokens);
  scheduler.schedule(request);
  scheduler.step(absl::Seconds(1));
  absl::SleepFor(absl::Seconds(2));
  auto sequence_result = request_wrapper.sequence_output();

  EXPECT_EQ(spec_steps, scheduler.get_ssm_execute_model_calls());
  EXPECT_EQ(0, scheduler.get_llm_execute_model_calls());
  EXPECT_EQ(0, scheduler.get_ssm_validate_calls());
  EXPECT_EQ(1, scheduler.get_llm_validate_calls());

  EXPECT_EQ(sequence_result.finish_reason, FinishReason::NONE);
  EXPECT_EQ(sequence_result.output_text, expect_output);
}

TEST(SpeculativeSchedulerTest, Speculative4StepsFullyMatchTest) {
  // who[1058] is[338] mess[4473] i[29875] ?[29973]
  const std::vector<int64_t> spec_token_ids = {1058, 338, 4473, 29875};
  const std::vector<int64_t> valid_token_ids = {1058, 338, 4473, 29875, 29973};
  const uint64_t spec_steps = 4;

  TestableSpeculativeScheduler scheduler(spec_token_ids, valid_token_ids,
      spec_steps);
  RequestWrapper request_wrapper(scheduler.tokenizer());

  std::vector<int> input_tokens = {1058};
  // expect output tokens: [1058] + [1058, 338, 4473, 29875, 29973] 
  // expect output string: who who is messi?
  const std::string expect_output("who who is messi?");
  auto request = request_wrapper.create_request(input_tokens);
  scheduler.schedule(request);
  scheduler.step(absl::Seconds(1));
  absl::SleepFor(absl::Seconds(2));
  auto sequence_result = request_wrapper.sequence_output();

  EXPECT_EQ(spec_steps, scheduler.get_ssm_execute_model_calls());
  EXPECT_EQ(0, scheduler.get_llm_execute_model_calls());
  EXPECT_EQ(0, scheduler.get_ssm_validate_calls());
  EXPECT_EQ(1, scheduler.get_llm_validate_calls());

  EXPECT_EQ(sequence_result.finish_reason, FinishReason::NONE);
  EXPECT_EQ(sequence_result.output_text, expect_output);
}

TEST(SpeculativeSchedulerTest, Speculative5StepsNoMatchTest) {
  // who[1058] is[338] mess[4473] i[29875] ?[29973]
  const std::vector<int64_t> spec_token_ids = {1058, 338, 4473, 29875, 29973};
  const std::vector<int64_t> valid_token_ids = {338, 4473, 29875, 29973, 1058, 338};
  const uint64_t spec_steps = 4;

  TestableSpeculativeScheduler scheduler(spec_token_ids, valid_token_ids,
      spec_steps);
  RequestWrapper request_wrapper(scheduler.tokenizer());

  std::vector<int> input_tokens = {1058};
  // expect output tokens: [1058] + [338] 
  // expect output string: who is
  const std::string expect_output("who is");
  auto request = request_wrapper.create_request(input_tokens);
  scheduler.schedule(request);
  scheduler.step(absl::Seconds(1));
  absl::SleepFor(absl::Seconds(2));
  auto sequence_result = request_wrapper.sequence_output();

  EXPECT_EQ(spec_steps, scheduler.get_ssm_execute_model_calls());
  EXPECT_EQ(0, scheduler.get_llm_execute_model_calls());
  EXPECT_EQ(0, scheduler.get_ssm_validate_calls());
  EXPECT_EQ(1, scheduler.get_llm_validate_calls());

  EXPECT_EQ(sequence_result.finish_reason, FinishReason::NONE);
  EXPECT_EQ(sequence_result.output_text, expect_output);
}

TEST(SpeculativeSchedulerTest, Speculative5StepsPartialMatchWithEndTest) {
  // who[1058] is[338] mess[4473] i[29875] ?[29973]
  const std::vector<int64_t> spec_token_ids = {1058, 338, 4473, 29875, 29973};
  const std::vector<int64_t> valid_token_ids = {1058, 338, 4473, 2};
  const uint64_t spec_steps = 5;

  TestableSpeculativeScheduler scheduler(spec_token_ids, valid_token_ids,
      spec_steps);
  RequestWrapper request_wrapper(scheduler.tokenizer());

  std::vector<int> input_tokens = {1058};
  // expect output tokens: [1058] + [1058, 338, 4473, 2] 
  // expect output string: who who is mess
  const std::string expect_output("who who is mess");
  auto request = request_wrapper.create_request(input_tokens);
  scheduler.schedule(request);
  scheduler.step(absl::Seconds(1));
  absl::SleepFor(absl::Seconds(2));
  auto sequence_result = request_wrapper.sequence_output();

  EXPECT_EQ(spec_steps, scheduler.get_ssm_execute_model_calls());
  EXPECT_EQ(0, scheduler.get_llm_execute_model_calls());
  EXPECT_EQ(0, scheduler.get_ssm_validate_calls());
  EXPECT_EQ(1, scheduler.get_llm_validate_calls());

  EXPECT_EQ(sequence_result.output_text, expect_output);
  EXPECT_EQ(sequence_result.finish_reason, FinishReason::STOP);
}

TEST(SpeculativeSchedulerTest, Speculative20StepsPartialMatchTest) {
  // who[1058] is[338] mess[4473] i[29875] ?[29973]
  const std::vector<int64_t> spec_token_ids = {1058, 338, 4473, 29875,
      1058, 338, 4473, 29875, 1058, 338, 4473, 29875, 1058, 338, 4473,
      29875, 1058, 338, 4473, 29875};
  const std::vector<int64_t> valid_token_ids = {1058, 338, 4473, 29875,
      1058, 338, 4473, 29875, 2};
  const uint64_t spec_steps = 20;

  TestableSpeculativeScheduler scheduler(spec_token_ids, valid_token_ids,
      spec_steps);
  RequestWrapper request_wrapper(scheduler.tokenizer());

  std::vector<int> input_tokens = {1058};
  // expect output tokens: [1058] +
  //   [1058, 338, 4473, 29875, 1058, 338, 4473, 29875, 2]
  // expect output string: who who is messi who is messi
  const std::string expect_output("who who is messi who is messi");
  auto request = request_wrapper.create_request(input_tokens);
  scheduler.schedule(request);
  scheduler.step(absl::Seconds(1));
  absl::SleepFor(absl::Seconds(2));
  auto sequence_result = request_wrapper.sequence_output();

  EXPECT_EQ(spec_steps, scheduler.get_ssm_execute_model_calls());
  EXPECT_EQ(0, scheduler.get_llm_execute_model_calls());
  EXPECT_EQ(0, scheduler.get_ssm_validate_calls());
  EXPECT_EQ(1, scheduler.get_llm_validate_calls());

  EXPECT_EQ(sequence_result.output_text, expect_output);
  EXPECT_EQ(sequence_result.finish_reason, FinishReason::STOP);
}
}  // namespace llm
