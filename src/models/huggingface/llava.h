#pragma once

#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include "chat_template/common_chat_template.h"
#include "layers/activation.h"
#include "layers/attention/attention.h"
#include "layers/attention/handler.h"
#include "layers/embedding.h"
#include "layers/linear.h"
#include "layers/normalization.h"
#include "layers/qkv_linear.h"
#include "memory/kv_cache.h"
#include "models/huggingface/llama.h"
#include "models/model_args.h"
#include "models/model_registry.h"
#include "models/parameters.h"

// llava model compatible with huggingface weights
namespace llm::hf {
class LlavaProjectorImpl : public torch::nn::Module {
 public:
  LlavaProjectorImpl(const ModelArgs& args,
                     const QuantArgs& quant_args,
                     const ParallelArgs& parallel_args,
                     const torch::TensorOptions& options) {
    act_func_ = Activation::get_act_func("gelu", options.device());
    CHECK(act_func_ != nullptr);

    auto int64_t vision_hidden_size = args.vision_hidden_size();
    auto int64_t text_hidden_size = args.text_hidden_size();
    // register the weight parameter
    linear_1_ = register_module("linear_1",
                                ColumnParallelLinear(vision_hidden_size,
                                                     text_hidden_size,
                                                     /*bias=*/true,
                                                     /*gather_output=*/false,
                                                     quant_args,
                                                     parallel_args,
                                                     options));
    linear_2_ =
        register_module("linear_2",
                        RowParallelLinear(text_hidden_size,
                                          text_hidden_size,
                                          /*bias=*/true,
                                          /*input_is_parallelized=*/true,
                                          quant_args,
                                          parallel_args,
                                          options));
  }

  torch::Tensor forward(torch::Tensor image_features) {
    return linear_2_(act_func_(linear_1(image_features)));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {}

  void verify_loaded_weights(const std::string& prefix) const {}

 private:
  ColumnParallelLinear linear_1_{nullptr};
  RowParallelLinear linear_2_{nullptr};

  ActFunc act_func_{nullptr};
};
TORCH_MODULE(LlavaProjector);

class CLIPVisionEmbeddingImpl : public torch::nn::Module {
 public:
  // int64_t hidden_size, int64_t image_size, int64_t patch_size, int64_t num_channels
  CLIPVisionEmbeddingImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options) {
    embed_dim_ = args.embed_dim;
    class_embedding_ = register_parameter("class_embedding",
        torch::randn({embed_dim_}));
    patch_embedding_ = register_module("patch_embedding",
        torch::nn::Conv2d(
          torch::nn::Conv2dOptions(args.num_channels, embed_dim_,
            args.patch_size).stride(args.patch_size).bias(false)));
    auto num_patches = (args.image_size / args.patch_size) *
                       (args.image_size / args.patch_size);
    auto num_positions = num_patches + 1;
    position_embedding_ = register_parameter("position_embedding",
        torch::randn({num_positions, embed_dim_}));
    position_ids = register_buffer("position_ids",
        torch::arange(0, num_positions, torch::kLong).unsqueeze(0));
  }

  torch::Tensor forward(const torch::Tensor& pixel_values) {
    int64_t batch_size = pixel_values.size(0);
    auto patch_embeds = patch_embedding_->forward(
        pixel_values.to(patch_embedding_->weight.dtype()))
                    .flatten(2)
                    .transpose(1, 2);

    auto class_embeds = class_embedding_.expand({batch_size, 1, embed_dim_});
    auto embeddings = torch::cat({class_embeds, patch_embeds}, 1);
    embeddings += position_embedding_.index({position_ids_});
    return embeddings;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {}

  void verify_loaded_weights() const {}

 private:
  int64_t embed_dim_;

  torch::Tensor class_embedding_;
  torch::Tensor position_ids_;
  torch::nn::Conv2d patch_embedding_{nullptr};
  torch::nn::Embedding position_embedding_{nullptr};
};
TORCH_MODULE(CLIPVisionEmbedding);

class CLIPMLPImpl : public torch::nn::Module {
 public:
  CLIPMLPImpl(const ModelArgs& args,
              const QuantArgs& quant_args,
              const ParallelArgs& parallel_args,
              const torch::TensorOptions& options) {
    // TODO: default activation is quick_gelu
    // https://github.com/huggingface/transformers/.../configuration_clip.py
    act_ = Activation::get_act_func(args.hidden_act(), options.device());
    CHECK(act_ != nullptr);

    fc1_ = register_module("fc1",
                           ColumnParallelLinear(args.hidden_size,
                                                args.itermediate_size,
                                                /*bias=*/true,
                                                /*gather_output=*/false,
                                                quant_args,
                                                parallel_args,
                                                options));
    fc2_ =register_module("fc2",
                          RowParallelLinear(args.intermediate_size,
                                            args.hidden_size,
                                            /*bias=*/true,
                                            /*input_is_parallelized*/true,
                                            quant_args,
                                            parallel_args,
                                            options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    return fc2_(act_(fc1_(hidden_states)));
  }

  void load_state_dict(const StateDict& state_dict) {}

  void verify_loaded_weights() const {}

 private:
  ActFunc act_{nullptr};
  torch::nn::Linear fc1_{nullptr};
  torch::nn::Linear fc2_{nullptr};
};
TORCH_MODULE(CLIPMLP);

// TODO: Optimize CLIPAttention
class CLIPAttentionImpl : public torch::nn::Module {
 public:
  CLIPAttentionImpl(const ModelArgs& args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options) {
    const int32_t world_size = parallel_args.world_size();
    CHECK(args.hidden_size % args.n_heads_== 0);
    const int64_t head_dim = args.hidden_size() / args.n_heads();
    const int64_t n_local_heads = n_heads / world_size;

    qkv_sizes_ = {n_local_heads * head_dim,
                  n_local_heads * head_dim,
                  n_local_heads * head_dim};

    scale_ = std::pow(head_dim_, -0.5);
    dropout_ = args.attention_dropout;

    // register submodules
    qkv_proj_ = register_module("qkv_proj",
                                QKVColumnParallelLinear(args.hidden_size(),
                                                        3 * args.n_heads() * head_dim,
                                                        head_dim,
                                                        /*bias=*/false,
                                                        /*gather_output=*/false,
                                                        quant_args,
                                                        parallel_args,
                                                        options));

    o_proj_ = register_module("o_proj",
                              RowParallelLinear(args.hidden_size,
                                                args.hidden_size,
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_state) {
    auto qkv = qkv_proj_(hidden_state).split(/*split_size=*/qkv_sizes_, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);
    // TODO output = attn();
    return o_proj_(output);
  }

  void load_state_dict(const StateDict& state_dict) {}

  void verify_loaded_weights() const {}

 private:
  int64_t embed_dim_;
  int64_t num_heads_;
  int64_t head_dim_;
  float scale_;
  float dropout_;
  QKVColumnParallelLinear qkv_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};
};
TORCH_MODULE(CLIPAttention);

class CLIPEncoderLayerImpl : public torch::nn::Module {
 public:
  CLIPEncoderLayerImpl(const ModelArgs& args,
                       const QuantArgs& quant_args,
                       const ParallelArgs& parallel_args,
                       const torch::TensorOptions& options) {
    self_attn_ = register_module("self_attn",
        CLIPAttention(args, quant_args, parallel_args, options));
    layer_norm1_ = register_module("layer_norm1", LayerNorm(args.hidden_size,
          args.layer_norm_eps, /*bias=*/true, options));
    layer_norm2_ = register_module("layer_norm2", LayerNorm(args.hidden_size,
          args.layer_norm_eps, /*bias=*/true, options));
  }

  // TODO: self_attn, attention_mask, causal_attention_mask
  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto residual = hidden_states;
    auto self_attn_output = self_attn_(layer_norm1(hidden_states)) + residual;
    residual = hidden_states;
    hidden_states = mlp_(layer_norm2_(hidden_states)) + residual;
    return hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {}

  void verify_loaded_weights() const {}

 private:
  CLIPAttention self_attn_{nullptr};
  torch::nn::LayerNorm layer_norm1_{nullptr};
  CLIPMLP mlp_{nullptr};
  torch::nn::LayerNorm layer_norm2_{nullptr};
};
TORCH_MODULE(CLIPEncoderLayer);

class CLIPEncoderImpl : public torch::nn::Module {
 public:
  CLIPEncoderImpl(const ModelArgs& args,
                  const QuantArgs& quant_args,
                  const ParallelArgs& parallel_args,
                  const torch::TensorOptions& options) {
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = CLIPEncoderLayer(
          args, quant_args, parallel_args, options);
      layers_.push_back(block);
      blocks_->push_back(block);
    }
  }

  // Output hidden states for all intermediate layers
  std::vector<torch::Tensor> forward(const torch::Tensor& embeddings) {
    std::vector<torch::Tensor> output_hidden_states;
    auto hidden_states = embeddings;
    for (size_t i = 0; i < layers_.size(); ++i) {
      output_hidden_states.emplace_back(hidden_states);
      auto& layer = layers_[i];
      hidden_states = layer(hidden_states);
    }
    output_hidden_states.emplace_back(hidden_states);
    return output_hidden_states;
  }

  void load_state_dict(const StateDict& state_dict) {}

  void verify_loaded_weights() const {}

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<LlamaDecoderLayer> layers_;
};
TORCH_MODULE(CLIPEncoder);

class CLIPVisionTransformerImpl : public torch::nn::Module {
 public:
  CLIPVisionTransformerImpl(const ModelArgs& args,
                            const QuantArgs& quant_args,
                            const ParallelArgs& parallel_args,
                            const torch::TensorOptions& options) {
    embeddings_ = register_module("embeddings",
        CLIPVisionEmbedding(args, quant_args, parallel_args, options);
    pre_layernorm_ = register_module("pre_layernorm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions(args.hidden_size)
            .eps(args.layer_norm_eps)));
    encoder_ = register_module("encoder",
        CLIPEncoder(args, quant_args,parallel_args, options));
    /*
    post_layernorm_ = register_module("post_layernorm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions(args.hidden_size)
            .eps(args.layer_norm_eps)));*/
  }

  std::vector<torch::Tensor> forward(const torch::Tensor& pixel_values) {
    auto hidden_states = embeddings_(pixel_values);
    hidden_states = pre_layernorm_(hidden_state);

    return encoder_(hidden_states);

    // TODO: no need for pooled output
    // auto pooled_output = encoder_outputs.slice(1,0,1).squeeze(1);
    // pooled_output = post_layernorm_(pooled_output);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {}

  void verify_loaded_weights() const {}
 
 private:
  CLIPVisionEmbedding embeddings_{nullptr};
  torch::nn::LayerNorm pre_layernorm_{nullptr};
  CLIPEncoder encoder_{nullptr};
};
TORCH_MODULE(CLIPVisionTranformer);

// TODO: CLIP, https://github.com/huggingface/transformers
class CLIPVisionModelImpl : public torch::nn::Module {
 public:
  CLIPVisionModelImpl(const ModelArgs& args,
                      const QuantArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      const torch::TensorOptions& options) {
    transformer_ = register_module("transformer", CLIPVisionTranformer(
          args, quant_args, parallel_args, options))
  }

  // return hidden_state (TODO support return: output_attention, return_dict)
  std::vector<torch::Tensor> forward(const torch::Tensor& images) {
    return transformer_(images);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {}

  void verify_loaded_weights() const {}

 private:
  CLIPVisionTranformer transformer_{nullptr};
};
TORCH_MODULE(CLIPVisionModel);

class LlamaBackboneImpl : public torch::nn::Module {
 public:
  LlamaBackboneImpl(const ModelArgs& args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options) {
    handler_ = AttentionHandler::create_handler_with_rope(
        args, /*interleaved=*/false, options);
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.n_layers());
    for (int32_t i = 0; i < args.n_layers(); i++) {
      auto block = LlamaDecoderLayer(
          args, quant_args, parallel_args, options, handler_.get());
      layers_.emplace_back(block);
      blocks_.emplace_back(block);
    }
    norm_ = register_module(
        "norm", RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
  }

  torch::Tensor forward(torch::Tensor embed_tokens,
                        torch::Tensor positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto h = embed_tokens;
    // TODO: set working space for attention handler
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      h = layer(h, positions, kv_caches[i], input_params);
    }
    return norm_(h);
  }

  void load_state_dict(const StateDict& state_dict) {
    for (int i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.select("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

 private:
  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<LlamaDecoderLayer> layers_;

  RMSNorm norm_{nullptr};
};
TORCH_MODULE(LlamaBackbone);

class LlavaModelForCausalLMImpl : public torch::nn::Module {
 public:
  LlavaModelForCasalLMImpl(const ModelArgs& args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options) {
    vision_feature_layer_ = args.vision_feature_layer;
    vision_model_ = register_module(
        "vision_model",
        CLIPVisionModel(args, quant_args, parallel_args, options));

    projector_ = register_module(
        "projector", LlavaProjector(args, quant_args, parallel_args, options));

    embed_tokens_ = register_module(
        "embed_tokens",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));

    llm_model_ = register_module(
        "llm_model", LlamaBackbone(args, quant_args, parallel_args, options));

    lm_head_ = register_module("lm_head",
                               ColumnParallelLinear(args.hidden_size(),
                                                    args.vocab_size(),
                                                    /*bias=*/false,
                                                    /*gather_output*/ true,
                                                    parallel_args,
                                                    options));
  }

  torch::Tensor forward(const torch::Tensor& images,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto& text_embedding = embed_tokens_(tokens);

    // TODO: filter last_hidden_states
    auto& hidden_states = vision_model_(images);
    // Only use the last hidden states
    auto& last_hidden_states = hidden_states[vision_feature_layer_];
    auto& vision_embedding = projector_(last_hidden_state);

    merge_text_vision_embeddings(text_embedding, vision_embedding, tokens);
    return llm_model_(text_embedding, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    auto h = hidden_states;
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    return lm_head_(h);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {}

  void verify_loaded_weights() const {}

 private:
  void merge_text_vision_embeddings(torch::Tensor& text_embedding,
                                    const torch::Tensor& vision_embedding,
                                    const torch::Tensor& token_ids) {
    // TODO: configure image_token_ids
    constexpr int32_t image_token_id = 512;

    auto mask = (token_ids == image_token_id);
    text_embedding.index_put_(
        {mask}, vision_embedding.view({-1, vision_embedding.size(-1)}));
  }

 private:
  CLIPVisionModel vision_model_{nullptr};
  LlavaProjector projector_{nullptr};

  ParallelEmbedding embed_tokens_{nullptr};
  LlamaBackbone llm_model_{nullptr};
  ColumnParallelLinear lm_head_{nullptr};

  int64_t vision_feature_layer_;
};
TORCH_MODULE(LlavaForCausalLM);

}  // namespace llm::hf
