#include "models_handler.h"

#include <string>

#include "models.pb.h"

namespace llm {

ModelsHandler::ModelsHandler(const std::string& model_id)
    : model_id_(model_id), created_(absl::ToUnixSeconds(absl::Now())) {}

grpc::Status ModelsHandler::List(grpc::ServerContext* /*context*/,
                                 const proto::ListRequest* /*request*/,
                                 proto::ListResponse* response) {
  auto* model_card = response->add_data();
  model_card->set_id(model_id_);
  model_card->set_created(created_);
  model_card->set_object("model");
  model_card->set_owned_by("scalellm");
  return grpc::Status::OK;
}

}  // namespace llm