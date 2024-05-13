#include "models_handler.h"

#include <string>

#include "models.pb.h"

namespace llm {

ModelsHandler::ModelsHandler(const std::vector<std::string>& models)
    : models_(models), created_(absl::ToUnixSeconds(absl::Now())) {}

grpc::Status ModelsHandler::List(grpc::ServerContext* /*context*/,
                                 const proto::ListRequest* /*request*/,
                                 proto::ListResponse* response) {
  for (const auto& model_id : models_) {
    auto* model_card = response->add_data();
    model_card->set_id(model_id);
    model_card->set_created(created_);
    model_card->set_object("model");
    model_card->set_owned_by("scalellm");
  }
  return grpc::Status::OK;
}

}  // namespace llm