#include "module.h"

#include <ATen/ops/equal.h>
#include <c10/core/Device.h>
#include <gtest/gtest.h>

namespace llm {

class ParametersModel : public Module {
 public:
  ParametersModel(bool bias = true) {
    // register some parameters
    weight_ = register_parameter("weight", torch::randn({16, 16}));
    if (bias) {
      bias_ = register_parameter("bias", torch::randn({32, 32}));
    }

    sharded_param_ = register_sharded_parameter("sharded_param",
                                                /*dim=*/0,
                                                /*rank=*/0,
                                                /*world_size=*/2,
                                                torch::randn({16, 32}));
    undefined_param_ = register_parameter("undefined_param", torch::Tensor());
  }

  torch::Tensor weight_;
  torch::Tensor bias_;

  torch::Tensor sharded_param_;
  torch::Tensor undefined_param_;
};

class SubModel : public Module {
 public:
  SubModel() {
    param1_ = register_parameter("param1", torch::randn({8, 8}));
    param2_ = register_parameter("param2", torch::randn({8, 16}));
    x_ = register_module("x", std::make_shared<ParametersModel>(/*bias=*/true));
    y_ =
        register_module("y", std::make_shared<ParametersModel>(/*bias=*/false));
  }

  torch::Tensor param1_;
  torch::Tensor param2_;
  std::shared_ptr<ParametersModel> x_;
  std::shared_ptr<ParametersModel> y_;
};

class Model : public Module {
 public:
  Model() {
    param1_ = register_parameter("param1", torch::randn({8, 8}));
    param2_ = register_parameter("param2", torch::randn({8, 16}));
    sub_model_ = register_module("submodel", std::make_shared<SubModel>());
  }

  torch::Tensor param1_;
  torch::Tensor param2_;
  std::shared_ptr<SubModel> sub_model_;
};

TEST(ModuleTest, Parameters) {
  std::unordered_map<std::string, torch::Tensor> dict = {
      {"weight", torch::randn({16, 16})},
      {"bias", torch::randn({32, 32})},
      {"param1", torch::randn({16, 16})},
      {"param2", torch::randn({16, 16})},
      {"sharded_param", torch::randn({32, 32})},
      {"undefined_param", torch::randn({32, 32})}};

  StateDict state_dict(dict);
  EXPECT_EQ(state_dict.size(), dict.size());

  ParametersModel model;
  EXPECT_EQ(model.load(state_dict), 3);

  EXPECT_TRUE(torch::equal(model.weight_, dict["weight"]));
  EXPECT_TRUE(torch::equal(model.bias_, dict["bias"]));
  EXPECT_TRUE(torch::equal(model.sharded_param_,
                           dict["sharded_param"].chunk(2, /*dim=*/0)[0]));
  EXPECT_FALSE(model.undefined_param_.defined());

  EXPECT_TRUE(model.verify());
}

TEST(ModuleTest, SubModules) {
  std::unordered_map<std::string, torch::Tensor> dict = {
      {"param1", torch::randn({8, 8})},
      {"param2", torch::randn({8, 16})},
      {"submodel.param1", torch::randn({8, 8})},
      {"submodel.param2", torch::randn({8, 16})},
      {"submodel.x.weight", torch::randn({16, 16})},
      {"submodel.x.bias", torch::randn({32, 32})},
      {"submodel.x.sharded_param", torch::randn({32, 32})},
      {"submodel.x.undefined_param", torch::randn({32, 32})},
      {"submodel.y.weight", torch::randn({16, 16})},
      {"submodel.y.bias", torch::randn({32, 32})},
      {"submodel.y.sharded_param", torch::randn({32, 32})},
      {"submodel.y.undefined_param", torch::randn({32, 32})}};

  StateDict state_dict(dict);
  EXPECT_EQ(state_dict.size(), dict.size());

  Model model;
  EXPECT_EQ(model.load(state_dict), 9);

  EXPECT_TRUE(torch::equal(model.param1_, dict["param1"]));
  EXPECT_TRUE(torch::equal(model.param2_, dict["param2"]));
  // submodel
  EXPECT_TRUE(torch::equal(model.sub_model_->param1_, dict["submodel.param1"]));
  EXPECT_TRUE(torch::equal(model.sub_model_->param2_, dict["submodel.param2"]));
  EXPECT_TRUE(
      torch::equal(model.sub_model_->x_->weight_, dict["submodel.x.weight"]));
  EXPECT_TRUE(
      torch::equal(model.sub_model_->x_->bias_, dict["submodel.x.bias"]));
  EXPECT_TRUE(
      torch::equal(model.sub_model_->x_->sharded_param_,
                   dict["submodel.x.sharded_param"].chunk(2, /*dim=*/0)[0]));
  EXPECT_FALSE(model.sub_model_->x_->undefined_param_.defined());
  EXPECT_TRUE(
      torch::equal(model.sub_model_->y_->weight_, dict["submodel.y.weight"]));
  EXPECT_FALSE(model.sub_model_->y_->bias_.defined());
  EXPECT_TRUE(
      torch::equal(model.sub_model_->y_->sharded_param_,
                   dict["submodel.y.sharded_param"].chunk(2, /*dim=*/0)[0]));
  EXPECT_FALSE(model.sub_model_->y_->undefined_param_.defined());

  EXPECT_TRUE(model.verify());
}

}  // namespace llm
