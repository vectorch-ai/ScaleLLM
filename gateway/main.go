package main

import (
	"context"
	"flag"
	"io"
	"net/http"

	"github.com/golang/glog"
	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	// importing generated stubs
	gw "gateway/proto"
)

var (
	// command-line options:
	// gRPC server endpoint
	grpcServerEndpoint = flag.String("grpc-server-endpoint", "localhost:8888", "gRPC server endpoint")
)

type EventSourceMarshaler struct {
	JSONPb runtime.JSONPb
}

func (c *EventSourceMarshaler) Marshal(v interface{}) ([]byte, error) { return c.JSONPb.Marshal(v) }
func (c *EventSourceMarshaler) Unmarshal(data []byte, v interface{}) error {
	return c.JSONPb.Unmarshal(data, v)
}
func (c *EventSourceMarshaler) NewDecoder(r io.Reader) runtime.Decoder { return c.JSONPb.NewDecoder(r) }
func (c *EventSourceMarshaler) NewEncoder(w io.Writer) runtime.Encoder { return c.JSONPb.NewEncoder(w) }

func (m *EventSourceMarshaler) ContentType(v interface{}) string {
	return "text/event-stream"
}

func run() error {
	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Register gRPC server endpoint
	// Note: Make sure the gRPC server is running properly and accessible
	mux := runtime.NewServeMux(runtime.WithMarshalerOption(
		"application/json",
		&EventSourceMarshaler{JSONPb: runtime.JSONPb{}}))
	opts := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	err := gw.RegisterCompletionHandlerFromEndpoint(ctx, mux, *grpcServerEndpoint, opts)
	if err != nil {
		return err
	}

	// Start HTTP server (and proxy calls to gRPC server endpoint)
	return http.ListenAndServe(":8080", mux)
}

func main() {
	flag.Parse()
	defer glog.Flush()

	if err := run(); err != nil {
		glog.Fatal(err)
	}
}
