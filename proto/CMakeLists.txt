include(grpc_proto_library)

grpc_proto_library(
  NAME
    completion
  SRCS
    completion.proto
    google/api/annotations.proto
    google/api/http.proto
    google/api/field_behavior.proto
    protoc-gen-openapiv2/options/annotations.proto
    protoc-gen-openapiv2/options/openapiv2.proto
)
