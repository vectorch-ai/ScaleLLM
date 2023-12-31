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
	grpcServerEndpoint = flag.String("grpc-server", "127.0.0.1:8888", "gRPC server endpoint")
	httpServerEndpoint = flag.String("http-server", "0.0.0.0:8080", "HTTP server endpoint")
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
	// register completion handler
	err := RegisterCompletionHandlerFromEndpoint(ctx, handler, *grpcServerEndpoint, opts)
	if err != nil {
		glog.Error("Failed to register completion handler from endpoint ", err)
		return err
	}
	glog.Info("Register grpc server at ", *grpcServerEndpoint)
	// register chat handler
	err = RegisterChatHandlerFromEndpoint(ctx, handler, *grpcServerEndpoint, opts)
	if err != nil {
		glog.Error("Failed to register chat handler from endpoint ", err)
		return err
	}
	// register models handler
	err = RegisterModelsHandlerFromEndpoint(ctx, handler, *grpcServerEndpoint, opts)
	if err != nil {
		glog.Error("Failed to register models handler from endpoint ", err)
		return err
	}

	// Start HTTP server (and proxy calls to gRPC server endpoint)
	glog.Info("Starting HTTP server at ", *httpServerEndpoint, " ...")
	return http.ListenAndServe(*httpServerEndpoint, handler)
}

func main() {
	flag.Parse()
	defer glog.Flush()

	if err := run(); err != nil {
		glog.Fatal(err)
	}
}
