include(grpc_proto_library)

grpc_proto_library(
  NAME
    completion
  SRCS
    common.proto
    completion.proto
    chat.proto
    models.proto
)
