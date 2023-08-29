#!/bin/bash -e

# running in the same directory as this script
cd "$(dirname "$0")"

# remove old generated code
rm -rf proto openapiv2

# create proto directory if not exists
mkdir -p proto openapiv2

# generate gRPC code
protoc -I ../proto \
  --go_out ./proto --go_opt paths=source_relative \
  --go-grpc_out ./proto --go-grpc_opt paths=source_relative \
  --grpc-gateway_out ./proto --grpc-gateway_opt logtostderr=true --grpc-gateway_opt paths=source_relative \
  --openapiv2_out ./openapiv2 --openapiv2_opt logtostderr=true \
  ../proto/completion.proto
