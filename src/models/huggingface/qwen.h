#pragma once

#include <torch/torch.h>

#include "layers/activation.h"
#include "layers/attention.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "layers/pos_embedding.h"
#include "memory/kv_cache.h"
#include "models/args.h"
#include "models/input_parameters.h"

// QWen model compatible with huggingface weights
// adopted from https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
namespace llm::hf {

class QWenMLPImpl : public torch::nn::Module {
 public:
  QWenMLPImpl(const ModelArgs& args,
              const QuantizationArgs& quant_args,
              const ParallelArgs& parallel_args,
              torch::ScalarType dtype,
              const torch::Device& device) {
    act_ = Activation::get("silu", device);
    CHECK(act_ != nullptr);

    const int64_t hidden_size = args.hidden_size();
    const int64_t intermediate_size = args.intermediate_size();

    // register the weight parameter
    w1_w2_proj_ = register_module("gate_up_proj",
                                  ColumnParallelLinear(hidden_size,
                                                       intermediate_size * 2,
                                                       /*bias=*/false,
                                                       /*gather_output=*/false,
                                                       quant_args,
                                                       parallel_args,
                                                       dtype,
                                                       device));
    c_proj_ = register_module("c_proj",
                              RowParallelLinear(intermediate_size,
                                                hidden_size,
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                dtype,
                                                device));
  }

  torch::Tensor forward(torch::Tensor x) {
    auto gate_up_proj = w1_w2_proj_(x);
    auto chunks = gate_up_proj.chunk(/*chunks=*/2, /*dim=*/-1);
    return c_proj_(chunks[0] * act_(chunks[1]));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    w1_w2_proj_->load_state_dict(state_dict, {"w1.", "w2."});
    c_proj_->load_state_dict(state_dict.select("c_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    w1_w2_proj_->verify_loaded_weights(prefix + "[w1,w2].");
    c_proj_->verify_loaded_weights(prefix + "c_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear w1_w2_proj_{nullptr};
  RowParallelLinear c_proj_{nullptr};

  ActFunc act_{nullptr};
};
TORCH_MODULE(QWenMLP);

class QWenAttentionImpl : public torch::nn::Module {
 public:
  QWenAttentionImpl(const ModelArgs& args,
                    const QuantizationArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    torch::ScalarType dtype,
                    const torch::Device& device) {
    const auto world_size = parallel_args.world_size();
    const int64_t hidden_size = args.hidden_size();
    const int64_t n_heads = args.n_heads();
    const int64_t head_dim = hidden_size / args.n_heads();
    const int64_t n_local_heads = n_heads / world_size;

    // register submodules
    c_attn_ = register_module("c_attn",
                              ColumnParallelLinear(hidden_size,
                                                   3 * hidden_size,
                                                   /*bias=*/true,
                                                   /*gather_output=*/false,
                                                   quant_args,
                                                   parallel_args,
                                                   dtype,
                                                   device));

    c_proj_ = register_module("c_proj",
                              RowParallelLinear(hidden_size,
                                                hidden_size,
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                dtype,
                                                device));

    // initialize attention
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    atten_ = register_module("atten",
                             AttentionWithRoPE(n_local_heads,
                                               n_local_heads,
                                               head_dim,
                                               scale,
                                               /*rotary_dim=*/head_dim,
                                               args.rope_scaling(),
                                               args.rope_theta(),
                                               args.max_position_embeddings(),
                                               /*interleaved=*/false,
                                               dtype,
                                               device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    const auto num_tokens = x.size(0);
    // (num_tokens, dim) x (dim, n_local_heads * head_dim)
    // => (num_tokens, n_local_heads * head_dim)
    auto qkv = c_attn_(x).chunk(/*chunks=*/3, /*dim=*/-1);
    // calculate attention, output: (num_tokens, n_local_heads * head_dim)
    auto output =
        atten_(qkv[0], qkv[1], qkv[2], positions, kv_cache, input_params);
    return c_proj_(output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    c_attn_->load_state_dict(state_dict.select("c_attn."));
    c_proj_->load_state_dict(state_dict.select("c_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    c_attn_->verify_loaded_weights(prefix + "c_attn.");
    c_proj_->verify_loaded_weights(prefix + "c_proj.");
  }

 private:
  // parameter members, must be registered
  ColumnParallelLinear c_attn_{nullptr};

  RowParallelLinear c_proj_{nullptr};

  // module members without parameters
  AttentionWithRoPE atten_{nullptr};
};
TORCH_MODULE(QWenAttention);

class QWenBlockImpl : public torch::nn::Module {
 public:
  QWenBlockImpl(const ModelArgs& args,
                const QuantizationArgs& quant_args,
                const ParallelArgs& parallel_args,
                torch::ScalarType dtype,
                const torch::Device& device) {
    // register submodules
    attn_ = register_module(
        "attn", QWenAttention(args, quant_args, parallel_args, dtype, device));
    mlp_ = register_module(
        "mlp", QWenMLP(args, quant_args, parallel_args, dtype, device));
    ln_1_ = register_module(
        "ln_1",
        RMSNorm(args.hidden_size(), args.rms_norm_eps(), dtype, device));
    ln_2_ = register_module(
        "ln_2",
        RMSNorm(args.hidden_size(), args.rms_norm_eps(), dtype, device));
  }

  torch::Tensor forward(torch::Tensor x,
                        torch::Tensor positions,
                        KVCache& kv_cache,
                        const InputParameters& input_params) {
    auto h = x + attn_(ln_1_(x), positions, kv_cache, input_params);
    return h + mlp_(ln_2_(h));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    attn_->load_state_dict(state_dict.select("attn."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    ln_1_->load_state_dict(state_dict.select("ln_1."));
    ln_2_->load_state_dict(state_dict.select("ln_2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    attn_->verify_loaded_weights(prefix + "attn.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    ln_1_->verify_loaded_weights(prefix + "ln_1.");
    ln_2_->verify_loaded_weights(prefix + "ln_2.");
  }

 private:
  // parameter members, must be registered
  QWenAttention attn_{nullptr};

  QWenMLP mlp_{nullptr};

  RMSNorm ln_1_{nullptr};

  RMSNorm ln_2_{nullptr};
};
TORCH_MODULE(QWenBlock);

class QWenModelImpl : public torch::nn::Module {
 public:
  QWenModelImpl(const ModelArgs& args,
                const QuantizationArgs& quant_args,
                const ParallelArgs& parallel_args,
                torch::ScalarType dtype,
                const torch::Device& device) {
    // register submodules
    wte_ = register_module("wte",
                           ParallelEmbedding(args.vocab_size(),
                                             args.hidden_size(),
                                             parallel_args,
                                             dtype,
                                             device));
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = QWenBlock(args, quant_args, parallel_args, dtype, device);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    ln_f_ = register_module(
        "ln_f",
        RMSNorm(args.hidden_size(), args.rms_norm_eps(), dtype, device));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = wte_(tokens);
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    return ln_f_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    wte_->load_state_dict(state_dict.select("wte."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(
          state_dict.select("h." + std::to_string(i) + "."));
    }
    ln_f_->load_state_dict(state_dict.select("ln_f."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    wte_->verify_loaded_weights(prefix + "wte.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "h." + std::to_string(i) +
                                        ".");
    }
    ln_f_->verify_loaded_weights(prefix + "ln_f.");
  }

 private:
  // parameter members, must be registered
  ParallelEmbedding wte_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<QWenBlock> layers_;

  RMSNorm ln_f_{nullptr};
};
TORCH_MODULE(QWenModel);

class QWenForCausalLMImpl : public torch::nn::Module {
 public:
  QWenForCausalLMImpl(const ModelArgs& args,
                      const QuantizationArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      torch::ScalarType dtype,
                      const torch::Device& device) {
    // register submodules
    transformer_ = register_module(
        "transformer",
        QWenModel(args, quant_args, parallel_args, dtype, device));

    lm_head_ = register_module("lm_head",
                               ColumnParallelLinear(args.hidden_size(),
                                                    args.vocab_size(),
                                                    /*bias=*/false,
                                                    /*gather_output=*/true,
                                                    parallel_args,
                                                    dtype,
                                                    device));
  }

  // tokens: [num_tokens]
  // positions: [num_tokens] token pos in the sequence
  torch::Tensor forward(torch::Tensor tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = transformer_(tokens, positions, kv_caches, input_params);
    // select last token for each sequence
    h = h.index_select(/*dim=*/0, input_params.last_token_indicies);
    return lm_head_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    transformer_->load_state_dict(state_dict.select("transformer."));
    lm_head_->load_state_dict(state_dict.select("lm_head."));
  }

  void verify_loaded_weights() const {
    transformer_->verify_loaded_weights("transformer.");
    lm_head_->verify_loaded_weights("lm_head.");
  }

 private:
  // parameter members, must be registered
  QWenModel transformer_{nullptr};

  ColumnParallelLinear lm_head_{nullptr};
};
TORCH_MODULE(QWenForCausalLM);

}  // namespace llm::hf
