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

// llava model: llava-hf/llava-1.5-7b-hf

namespace llm::hf {
class CLIPVisionEmbeddingImpl : public torch::nn::Module {
 public:
  CLIPVisionEmbeddingImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options) {
    embed_dim_ = args.mm_hidden_size();
    class_embedding_ =
        register_parameter("class_embedding", torch::randn({embed_dim_}));
    patch_embedding_ = register_module(
        "patch_embedding",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(args.mm_num_channels(),
                                                   embed_dim_,
                                                   args.mm_patch_size())
                              .stride(args.mm_patch_size())
                              .bias(false)));
    auto num_patches = (args.mm_image_size() / args.mm_patch_size()) *
                       (args.mm_image_size() / args.mm_patch_size());
    auto num_positions = num_patches + 1;
    position_embedding_ = register_parameter(
        "position_embedding", torch::randn({num_positions, embed_dim_}));
    position_ids_ = register_buffer(
        "position_ids",
        torch::arange(0, num_positions, torch::kLong).unsqueeze(0));
  }

  torch::Tensor forward(const torch::Tensor& pixel_values) {
    int64_t batch_size = pixel_values.size(0);
    auto patch_embeds =
        patch_embedding_
            ->forward(pixel_values.to(patch_embedding_->weight.dtype()))
            .flatten(2)
            .transpose(1, 2);

    auto class_embeds = class_embedding_.expand({batch_size, 1, embed_dim_});
    auto embeddings = torch::cat({class_embeds, patch_embeds}, 1);
    embeddings += position_embedding_.index({position_ids_});
    return embeddings;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    const auto cls = state_dict.get_tensor("class_embedding");
    DCHECK_EQ(cls.sizes(), class_embedding_.sizes());
    class_embedding_.copy_(cls);

    const auto pos = state_dict.get_tensor("position_embedding.weight");
    DCHECK_EQ(pos.sizes(), position_embedding_.sizes());
    position_embedding_.copy_(pos);

    const auto weight = state_dict.get_tensor("patch_embedding.weights");
    DCHECK_EQ(patch_embedding_->weight.sizes(), weight.sizes());
    patch_embedding_->weight.copy_(weight);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    // No need to verify, already checked in load_state_dict
  }

 private:
  int64_t embed_dim_;

  torch::Tensor class_embedding_;
  torch::Tensor position_ids_;
  torch::nn::Conv2d patch_embedding_{nullptr};
  torch::Tensor position_embedding_{nullptr};
};
TORCH_MODULE(CLIPVisionEmbedding);

class CLIPMLPImpl : public torch::nn::Module {
 public:
  CLIPMLPImpl(const ModelArgs& args,
              const QuantArgs& quant_args,
              const ParallelArgs& parallel_args,
              const torch::TensorOptions& options) {
    // TODO: default activation is quick_gelu, need to support quick_gelu
    // https://github.com/huggingface/transformers/.../configuration_clip.py
    act_ = Activation::get_act_func(args.mm_hidden_act(), options.device());
    CHECK(act_ != nullptr);

    fc1_ = register_module("fc1",
                           ColumnParallelLinear(args.mm_hidden_size(),
                                                args.mm_intermediate_size(),
                                                /*bias=*/true,
                                                /*gather_output=*/false,
                                                quant_args,
                                                parallel_args,
                                                options));
    fc2_ = register_module("fc2",
                           RowParallelLinear(args.mm_intermediate_size(),
                                             args.mm_hidden_size(),
                                             /*bias=*/true,
                                             /*input_is_parallelized*/ true,
                                             quant_args,
                                             parallel_args,
                                             options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    return fc2_(act_(fc1_(hidden_states)));
  }

  void load_state_dict(const StateDict& state_dict) {
    fc1_->load_state_dict(state_dict.select("fc1."));
    fc2_->load_state_dict(state_dict.select("fc2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    fc1_->verify_loaded_weights(prefix + "fc1.");
    fc2_->verify_loaded_weights(prefix + "fc2.");
  }

 private:
  ActFunc act_{nullptr};
  ColumnParallelLinear fc1_{nullptr};
  RowParallelLinear fc2_{nullptr};
};
TORCH_MODULE(CLIPMLP);

// TODO: Optimize CLIPAttention
class CLIPAttentionImpl : public torch::nn::Module {
 public:
  CLIPAttentionImpl(const ModelArgs& args,
                    const QuantArgs& quant_args,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options) {
    CHECK(args.mm_hidden_size() % args.mm_num_attention_heads() == 0);

    head_dim_ = args.mm_head_dim();
    embed_dim_ = args.mm_hidden_size();
    const int32_t world_size = parallel_args.world_size();
    num_heads_ = args.mm_num_attention_heads();
    const int64_t n_local_heads = num_heads_ / world_size;

    qkv_sizes_ = {n_local_heads * args.mm_head_dim(),
                  n_local_heads * args.mm_head_dim(),
                  n_local_heads * args.mm_head_dim()};

    scale_ = 1.0f / std::sqrt(static_cast<float>(args.mm_head_dim()));
    dropout_ = args.mm_dropout();

    // register submodules
    qkv_proj_ = register_module("qkv_proj",
                                QKVColumnParallelLinear(args.mm_hidden_size(),
                                                        num_heads_,
                                                        num_heads_,
                                                        head_dim_,
                                                        /*bias=*/false,
                                                        /*gather_output=*/false,
                                                        quant_args,
                                                        parallel_args,
                                                        options));

    o_proj_ = register_module("o_proj",
                              RowParallelLinear(args.mm_hidden_size(),
                                                args.mm_hidden_size(),
                                                /*bias=*/false,
                                                /*input_is_parallelized=*/true,
                                                quant_args,
                                                parallel_args,
                                                options));
  }

  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto qkv =
        qkv_proj_(hidden_states).split(/*split_size=*/qkv_sizes_, /*dim=*/-1);
    DCHECK_EQ(qkv.size(), 3);

    auto query_states = qkv[0] * scale_;
    auto bsz = hidden_states.size(0);
    auto tgz_len = hidden_states.size(1);
    auto key_states = shape(qkv[1], -1, bsz);
    auto value_states = shape(qkv[2], -1, bsz);

    auto proj_shape = std::vector<int64_t>{bsz * num_heads_, -1, head_dim_};
    query_states = shape(query_states, tgz_len, bsz).view(proj_shape);
    key_states = key_states.view(proj_shape);
    value_states = value_states.view(proj_shape);

    auto src_len = key_states.size(1);
    auto attn_weights = torch::bmm(query_states, key_states.transpose(1, 2));
    DCHECK_EQ(attn_weights.sizes(),
              torch::IntArrayRef({bsz * num_heads_, tgz_len, src_len}));

    attn_weights = torch::softmax(attn_weights, -1);
    auto attn_probs = torch::dropout(attn_weights, dropout_, false);
    auto attn_output = torch::bmm(attn_probs, value_states);

    DCHECK_EQ(attn_output.sizes(),
              torch::IntArrayRef({bsz * num_heads_, tgz_len, head_dim_}));
    attn_output =
        attn_output
            .view(torch::IntArrayRef({bsz, num_heads_, tgz_len, head_dim_}))
            .transpose(1, 2)
            .contiguous();
    attn_output =
        attn_output.view(torch::IntArrayRef({bsz, tgz_len, embed_dim_}));

    return o_proj_(attn_output);
  }

  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    qkv_proj_->load_state_dict(
        state_dict, {"q_proj.", "k_proj.", "v_proj."}, {"k_proj.", "v_proj."});
    o_proj_->load_state_dict(state_dict.select("out_proj."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    qkv_proj_->verify_loaded_weights(prefix + "[q_proj,k_proj,v_proj].");
    o_proj_->verify_loaded_weights(prefix + "out_proj.");
  }

 private:
  torch::Tensor shape(torch::Tensor tensor, int64_t seq_len, int64_t bsz) {
    return tensor.view({bsz, seq_len, num_heads_, head_dim_})
        .transpose(1, 2)
        .contiguous();
  }

 private:
  int64_t embed_dim_;
  int64_t num_heads_;
  int64_t head_dim_;
  float scale_;
  float dropout_;
  std::vector<int64_t> qkv_sizes_;
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
    self_attn_ = register_module(
        "self_attn", CLIPAttention(args, quant_args, parallel_args, options));
    layer_norm1_ = register_module("layer_norm1",
                                   LayerNorm(args.mm_hidden_size(),
                                             args.mm_layer_norm_eps(),
                                             /*bias=*/true,
                                             options));
    layer_norm2_ = register_module("layer_norm2",
                                   LayerNorm(args.mm_hidden_size(),
                                             args.mm_layer_norm_eps(),
                                             /*bias=*/true,
                                             options));
  }

  // TODO: self_attn, attention_mask, causal_attention_mask
  torch::Tensor forward(const torch::Tensor& hidden_states) {
    auto residual = hidden_states;
    auto h = self_attn_(layer_norm1_(hidden_states)) + residual;
    residual = h;
    h = mlp_(layer_norm2_(h)) + residual;
    return h;
  }

  void load_state_dict(const StateDict& state_dict) {
    self_attn_->load_state_dict(state_dict.select("self_attn."));
    layer_norm1_->load_state_dict(state_dict.select("layer_norm1."));
    mlp_->load_state_dict(state_dict.select("mlp."));
    layer_norm2_->load_state_dict(state_dict.select("layer_norm2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    self_attn_->verify_loaded_weights(prefix + "self_attn.");
    layer_norm1_->verify_loaded_weights(prefix + "layer_norm1.");
    mlp_->verify_loaded_weights(prefix + "mlp.");
    layer_norm2_->verify_loaded_weights(prefix + "layer_norm2.");
  }

 private:
  CLIPAttention self_attn_{nullptr};
  LayerNorm layer_norm1_{nullptr};
  CLIPMLP mlp_{nullptr};
  LayerNorm layer_norm2_{nullptr};
};
TORCH_MODULE(CLIPEncoderLayer);

class CLIPEncoderImpl : public torch::nn::Module {
 public:
  CLIPEncoderImpl(const ModelArgs& args,
                  const QuantArgs& quant_args,
                  const ParallelArgs& parallel_args,
                  const torch::TensorOptions& options) {
    layers_.reserve(args.mm_num_hidden_layers());
    for (int32_t i = 0; i < args.mm_num_hidden_layers(); i++) {
      auto block = CLIPEncoderLayer(args, quant_args, parallel_args, options);
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

  void load_state_dict(const StateDict& state_dict) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->load_state_dict(
          state_dict.select("layers." + std::to_string(i) + "."));
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
  }

 private:
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<CLIPEncoderLayer> layers_;
};
TORCH_MODULE(CLIPEncoder);

class CLIPVisionTransformerImpl : public torch::nn::Module {
 public:
  CLIPVisionTransformerImpl(const ModelArgs& args,
                            const QuantArgs& quant_args,
                            const ParallelArgs& parallel_args,
                            const torch::TensorOptions& options) {
    embeddings_ = register_module(
        "embeddings",
        CLIPVisionEmbedding(args, quant_args, parallel_args, options));
    pre_layernorm_ = register_module(
        "pre_layernorm",
        LayerNorm(args.mm_hidden_size(), args.layer_norm_eps(), true, options));
    encoder_ = register_module(
        "encoder", CLIPEncoder(args, quant_args, parallel_args, options));
    post_layernorm_ = register_module(
        "post_layernorm",
        LayerNorm(args.mm_hidden_size(), args.layer_norm_eps(), true, options));
  }

  std::vector<torch::Tensor> forward(const torch::Tensor& pixel_values) {
    auto hidden_states = embeddings_(pixel_values);
    hidden_states = pre_layernorm_(hidden_states);

    auto encoder_output = encoder_(hidden_states);
    // when return_dict = False, skip pooled output step.
    // auto pooled_output = encoder_outputs.slice(1,0,1).squeeze(1);
    // pooled_ouput = post_layernorm_(pooled_output);*/
    return encoder_output;
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    embeddings_->load_state_dict(state_dict.select("embeddings."));
    pre_layernorm_->load_state_dict(state_dict.select("pre_layrnorm."));
    encoder_->load_state_dict(state_dict.select("encoder."));
    post_layernorm_->load_state_dict(state_dict.select("post_layernorm."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    embeddings_->verify_loaded_weights(prefix + "embeddings.");
    pre_layernorm_->verify_loaded_weights(prefix + "pre_layrnorm.");
    encoder_->verify_loaded_weights(prefix + "encoder.");
    post_layernorm_->verify_loaded_weights(prefix + "post_layernorm.");
  }

 private:
  CLIPVisionEmbedding embeddings_{nullptr};
  LayerNorm pre_layernorm_{nullptr};
  CLIPEncoder encoder_{nullptr};
  LayerNorm post_layernorm_{nullptr};
};
TORCH_MODULE(CLIPVisionTransformer);

// Follow implementation: https://github.com/huggingface/transformers
class CLIPVisionModelImpl : public torch::nn::Module {
 public:
  CLIPVisionModelImpl(const ModelArgs& args,
                      const QuantArgs& quant_args,
                      const ParallelArgs& parallel_args,
                      const torch::TensorOptions& options) {
    transformer_ = register_module(
        "transformer",
        CLIPVisionTransformer(args, quant_args, parallel_args, options));
  }

  // return hidden_state (TODO support return: output_attention, return_dict)
  std::vector<torch::Tensor> forward(const torch::Tensor& images) {
    return transformer_(images);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    transformer_->load_state_dict(state_dict.select("vision_model."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    transformer_->verify_loaded_weights(prefix + "vision_model.");
  }

 private:
  CLIPVisionTransformer transformer_{nullptr};
};
TORCH_MODULE(CLIPVisionModel);

// Not used but need to support
class LlavaProjectorLinearImpl : public torch::nn::Module {
 public:
  LlavaProjectorLinearImpl(const ModelArgs& args,
                           const QuantArgs& quant_args,
                           const ParallelArgs& parallel_args,
                           const torch::TensorOptions& options) {
    linear_ = register_module("linear",
                              ColumnParallelLinear(args.mm_hidden_size(),
                                                   args.hidden_size(),
                                                   /*bias=*/true,
                                                   /*gather_output=*/false,
                                                   quant_args,
                                                   parallel_args,
                                                   options));
  }

  torch::Tensor forward(torch::Tensor image_features) {
    return linear_(image_features);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    linear_->load_state_dict(state_dict.select("0."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_->verify_loaded_weights(prefix + "0.");
  }

 private:
  ColumnParallelLinear linear_{nullptr};
};
TORCH_MODULE(LlavaProjectorLinear);

class LlavaProjectorMLP2XImpl : public torch::nn::Module {
 public:
  LlavaProjectorMLP2XImpl(const ModelArgs& args,
                          const QuantArgs& quant_args,
                          const ParallelArgs& parallel_args,
                          const torch::TensorOptions& options) {
    linear_1_ = register_module("linear_0",
                                ColumnParallelLinear(args.mm_hidden_size(),
                                                     args.hidden_size(),
                                                     /*bias=*/true,
                                                     /*gather_output*/ false,
                                                     quant_args,
                                                     parallel_args,
                                                     options));
    linear_2_ = register_module("linear_2",
                                RowParallelLinear(args.hidden_size(),
                                                  args.hidden_size(),
                                                  /*bias=*/true,
                                                  /*gather_output*/ false,
                                                  quant_args,
                                                  parallel_args,
                                                  options));
    // projector's activation type is "gelu"
    act_ = Activation::get_act_func(args.mm_projector_hidden_act(),
                                    options.device());
    CHECK(act_ != nullptr);
  }

  torch::Tensor forward(torch::Tensor image_features) {
    return linear_2_(act_(linear_1_(image_features)));
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    linear_1_->load_state_dict(state_dict.select("linear_1."));
    linear_2_->load_state_dict(state_dict.select("linear_2."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    linear_1_->verify_loaded_weights(prefix + "linear_1.");
    linear_2_->verify_loaded_weights(prefix + "linear_2.");
  }

 private:
  ColumnParallelLinear linear_1_{nullptr};
  RowParallelLinear linear_2_{nullptr};
  ActFunc act_{nullptr};
};
TORCH_MODULE(LlavaProjectorMLP2X);

class LlavaModelImpl : public torch::nn::Module {
 public:
  LlavaModelImpl(const ModelArgs& args,
                 const QuantArgs& quant_args,
                 const ParallelArgs& parallel_args,
                 const torch::TensorOptions& options) {
    vision_feature_layer_ = args.mm_vision_feature_layer();
    vision_model_ = register_module(
        "vision_model",
        CLIPVisionModel(args, quant_args, parallel_args, options));

    projector_ = register_module(
        "projector",
        LlavaProjectorMLP2X(args, quant_args, parallel_args, options));

    embed_tokens_ = register_module(
        "embed_tokens",
        ParallelEmbedding(
            args.vocab_size(), args.hidden_size(), parallel_args, options));
    handler_ = AttentionHandler::create_handler_with_rope(
        args, /*interleaved=*/false, options);
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(args.mm_num_hidden_layers());
    for (int32_t i = 0; i < args.mm_num_hidden_layers(); i++) {
      auto block = LlamaDecoderLayer(
          args, quant_args, parallel_args, options, handler_.get());
      layers_.push_back(block);
      blocks_->push_back(block);
    }
    norm_ = register_module(
        "norm", RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
  }

  torch::Tensor vision_encode(const torch::Tensor& image,
                              const torch::Tensor& tokens) {
    auto text_embedding = embed_tokens_(tokens);
    // TODO: filter last_hidden_states
    const auto& image_hidden_states = vision_model_(image);
    // Only use the last hidden states
    const auto& last_hidden_states = image_hidden_states[vision_feature_layer_];
    const auto& vision_embedding = projector_(last_hidden_states);
    merge_text_vision_embeddings(text_embedding, vision_embedding, tokens);
    return text_embedding;
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    auto input_embedding = input_params.input_embedding;
    torch::Tensor hidden_states;
    if (!input_embedding.defined()) {
      hidden_states = embed_tokens_(tokens);
    } else {
      hidden_states = input_embedding;
    }
    // TODO: set working space for attention handler
    for (size_t i = 0; i < layers_.size(); i++) {
      auto& layer = layers_[i];
      hidden_states =
          layer(hidden_states, positions, kv_caches[i], input_params);
    }
    return norm_(hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    embed_tokens_->load_state_dict(
        state_dict.select("language_model.model.embed_tokens."));
    projector_->load_state_dict(state_dict.select("multi_modal_projector."));
    vision_model_->load_state_dict(state_dict.select("vision_tower."));
    // call each layer's load_state_dict function
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->load_state_dict(state_dict.select(
          "language_model.model.layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.select("language_model.model.norm."));
  }

  void verify_loaded_weights() const {
    embed_tokens_->verify_loaded_weights("language_model.model.embed_tokens.");
    projector_->verify_loaded_weights("multi_modal_projector.");
    vision_model_->verify_loaded_weights("vision_tower.");
    for (int i = 0; i < layers_.size(); i++) {
      layers_[i]->verify_loaded_weights("language_model.model.layers." +
                                        std::to_string(i) + ".");
    }
    norm_->verify_loaded_weights("language_model.model.norm.");
  }

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
  int64_t vision_feature_layer_;

  LlavaProjectorMLP2X projector_{nullptr};

  ParallelEmbedding embed_tokens_{nullptr};
  // attention handler
  std::unique_ptr<AttentionHandler> handler_{nullptr};

  torch::nn::ModuleList blocks_{nullptr};
  // hold same data but different type as blocks_ to avoid type cast
  std::vector<LlamaDecoderLayer> layers_;

  RMSNorm norm_{nullptr};
};
TORCH_MODULE(LlavaModel);

class LlavaForCausalVLMImpl : public torch::nn::Module {
 public:
  LlavaForCausalVLMImpl(const ModelArgs& args,
                        const QuantArgs& quant_args,
                        const ParallelArgs& parallel_args,
                        const torch::TensorOptions& options) {
    model_ = register_module(
        "model", LlavaModel(args, quant_args, parallel_args, options));

    lm_head_ = register_module("lm_head",
                               ColumnParallelLinear(args.hidden_size(),
                                                    args.vocab_size(),
                                                    /*bias=*/false,
                                                    /*gather_output*/ true,
                                                    parallel_args,
                                                    options));
  }

  torch::Tensor vision_encode(const torch::Tensor& image,
                              const torch::Tensor& tokens) {
    return model_->vision_encode(image, tokens);
  }

  // images is stored in input_params
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const InputParameters& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
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
  void load_state_dict(const StateDict& state_dict) {
    model_->load_state_dict(state_dict);
    lm_head_->load_state_dict(state_dict.select("language_model.lm_head."));
  }

  void verify_loaded_weights() const {
    model_->verify_loaded_weights();
    lm_head_->verify_loaded_weights("language_model.lm_head.");
  }

 private:
  LlavaModel model_{nullptr};
  ColumnParallelLinear lm_head_{nullptr};
};
TORCH_MODULE(LlavaForCausalVLM);

REGISTER_CAUSAL_VLM_MODEL(llava, LlavaForCausalVLM);

// REGISTER_DEFAULT_CHAT_TEMPLATE(llama, Llama2ChatTemplate);

REGISTER_MODEL_ARGS(llava, [&] {
  // vision config
  LOAD_ARG_OR(model_type, "model_type", "llava");
  LOAD_ARG_OR(mm_dropout, "vision_config.dropout", 0.0f);
  LOAD_ARG_OR(mm_hidden_act, "vision_config.hidden_act", "quick_gelu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.hidden_size", 1024);
  LOAD_ARG_OR(mm_image_size, "vision_config.image_size", 336);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.intermediate_size", 4096);
  LOAD_ARG_OR(mm_num_channels, "vision_config.num_channels", 3);
  LOAD_ARG_OR(mm_initializer_range, "vision_config.initializer_range", 0.02f);
  LOAD_ARG_OR(mm_layer_norm_eps, "vision_config.layer_norm_eps", 1e-05);
  LOAD_ARG_OR(mm_num_attention_heads, "vision_config.num_attention_heads", 12);
  LOAD_ARG_OR(mm_num_beam_groups, "vision_config.num_beam_groups", 1);
  LOAD_ARG_OR(mm_num_beams, "vision_config.num_beams", 1);
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.num_hidden_layers", 12);
  LOAD_ARG_OR(mm_num_return_sequences, "vision_config.num_return_sequences", 1);
  LOAD_ARG_OR(mm_output_attentions, "vision_config.output_attentions", false);
  LOAD_ARG_OR(
      mm_output_hidden_states, "vision_config.output_hidden_states", false);
  LOAD_ARG_OR(mm_output_scores, "vision_config.output_scores", false);
  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 32);
  LOAD_ARG_OR(mm_projection_dim, "vision_config.projection_dim", 512);
  LOAD_ARG_OR(
      mm_remove_invalid_values, "vision_config.remove_invalid_values", false);
  LOAD_ARG_OR(mm_repetition_penalty, "vision_config.repetition_penalty", 1.0f);
  LOAD_ARG_OR(mm_return_dict, "vision_config.return_dict", true);
  LOAD_ARG_OR(mm_return_dict_in_generate,
              "vision_config.return_dict_in_generate",
              false);
  LOAD_ARG_OR(mm_temperature, "vision_config.temperature", 1.0f);
  LOAD_ARG_OR(
      mm_tie_encoder_decoder, "vision_config.tie_encoder_decoder", false);
  LOAD_ARG_OR(
      mm_tie_word_embeddings, "vision_config.tie_word_embeddings", true);
  LOAD_ARG_OR(mm_top_k, "vision_config.top_k", 50);
  LOAD_ARG_OR(mm_top_p, "vision_config.top_p", 1.0f);
  LOAD_ARG_OR(mm_torchscript, "vision_config.torchscript", false);
  LOAD_ARG_OR(mm_use_bfloat16, "vision_config.user_bfloat16", false);
  LOAD_ARG_OR_FUNC(mm_head_dim, "mm_head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });
  LOAD_ARG_OR(mm_vocab_size, "vision_config.vocab_size", 32000);
  // projector config
  LOAD_ARG_OR(mm_projector_type, "mm_projector_type", "mlp2x_gelu");
  LOAD_ARG_OR(mm_projector_hidden_act, "projector_hidden_act", "gelu");
  LOAD_ARG_OR(mm_projector_n_layers, "mm_projector_n_layers", 2);
  LOAD_ARG_OR(mm_vision_feature_layer, "vision_feature_layer", -2);
  LOAD_ARG_OR(mm_vision_feature_select_strategy,
              "vision_feature_select_strategy",
              "default");
  // text config
  LOAD_ARG_OR(hidden_size, "hidden_size", 4096);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 32);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 32);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 11008);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 4096);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 1);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 2);
  LOAD_ARG_OR(rope_theta, "rope_theta", 10000.0f);
  LOAD_ARG_OR(rope_scaling, "rope_scaling", 1.0f);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 32064);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 32);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
});
}  // namespace llm::hf
