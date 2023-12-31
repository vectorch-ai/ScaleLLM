#!/bin/bash -e

# running in the same directory as this script
cd "$(dirname "$0")"

# remove old generated code
rm -rf proto

# create proto directory if not exists
mkdir -p proto

# generate gRPC and gateway code
protoc -I ../proto \
  --go_out ./proto \
    --go_opt paths=source_relative \
  --go-grpc_out ./proto \
    --go-grpc_opt paths=source_relative \
  ../proto/*.proto
