include(grpc_proto_library)

grpc_proto_library(
  NAME
    completion
  SRCS
    completion.proto
    google/api/annotations.proto
    google/api/http.proto
)
