package main

import (
	"context"
	"flag"
	"net/http"

	"github.com/golang/glog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	gw "github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
)

var (
	// command-line options:
	// gRPC server endpoint
	grpcServerEndpoint = flag.String("grpc-server-endpoint", "localhost:8888", "gRPC server endpoint")
)

func run() error {
	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Register gRPC server endpoint
	// Note: Make sure the gRPC server is running properly and accessible
	handler := NewHttpHandler(&gw.JSONPb{})
	// TODO: add TLS credentials
	opts := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	err := RegisterCompletionHandlerFromEndpoint(ctx, handler, *grpcServerEndpoint, opts)
	if err != nil {
		glog.Error("Failed to register handler from endpoint ", err)
		return err
	}

	// Start HTTP server (and proxy calls to gRPC server endpoint)
	glog.Info("Starting HTTP server at port 8080")
	return http.ListenAndServe(":8080", handler)
}

func main() {
	flag.Parse()
	defer glog.Flush()

	if err := run(); err != nil {
		glog.Fatal(err)
	}
}
