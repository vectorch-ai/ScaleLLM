The gRPC-Gateway is a plugin of the Google protocol buffers compiler
[protoc](https://github.com/protocolbuffers/protobuf).
It reads protobuf service definitions and generates a reverse-proxy server which
translates a RESTful HTTP API into gRPC. This server is generated according to the
[`google.api.http`](https://github.com/googleapis/googleapis/blob/master/google/api/http.proto#L46)
annotations in your service definitions.

This helps you provide your APIs in both gRPC and RESTful style at the same time.
<div align="center">
<img src="https://raw.githubusercontent.com/grpc-ecosystem/grpc-gateway/main/docs/assets/images/architecture_introduction_diagram.svg" />
</div>

## Prerequisites

### Install go environment [reference](https://golang.org/doc/install)

```sh
$ sudo apt install golang-go
$ go version
go version go1.18.1 linux/amd64
```

### Set the compilation environment

Add "$GOPATH/bin" to your "$PATH" environment variable so that the protoc compiler can find the plugins. You can check current GOPATH by running `go env GOPATH`.

```sh
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin
```

### Install dependencies

```sh
go get ./...
```

# Run the server

```sh
go run main.go
```

# Generate gRPC-Gateway files (optional)

You only need to install protoc if you want to recompile the `.proto` files.

## Protoc Installation [reference](https://grpc-ecosystem.github.io/grpc-gateway/docs/usage.html#installation)

Run `go mod tidy` to resolve the versions. Install by running

```sh
$ go install \
    github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-grpc-gateway \
    github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-openapiv2 \
    google.golang.org/protobuf/cmd/protoc-gen-go \
    google.golang.org/grpc/cmd/protoc-gen-go-grpc
```

This will place four binaries in your `$GOBIN`;

- `protoc-gen-grpc-gateway`
- `protoc-gen-openapiv2`
- `protoc-gen-go`
- `protoc-gen-go-grpc`

## Generate gRPC-Gateway files

you can regenerate the files by running the following command:

```sh
./generate.sh
```

this will generate the following files:

- `./proto/completion.pb.go`
- `./proto/completion_grpc.pb.go`
- `./proto/completion.pb.gw.go`
- `./openapiv2/completion.swagger.json`
