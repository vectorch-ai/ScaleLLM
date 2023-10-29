package main

import (
	"context"
	"io"
	"net/http"

	"github.com/golang/glog"
	"github.com/grpc-ecosystem/grpc-gateway/v2/utilities"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	// importing generated stubs
	scalellm "gateway/proto"

	gw "github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
)

func SendChateRequest(ctx context.Context, marshaler gw.Marshaler, client scalellm.CompletionClient, req *http.Request) (scalellm.Completion_CompleteClient, bool, error) {
	var protoReq scalellm.CompletionRequest
	newReader, berr := utilities.IOReaderFactory(req.Body)
	if berr != nil {
		return nil, false, status.Errorf(codes.InvalidArgument, "%v", berr)
	}
	if err := marshaler.NewDecoder(newReader()).Decode(&protoReq); err != nil && err != io.EOF {
		return nil, false, status.Errorf(codes.InvalidArgument, "%v", err)
	}

	stream, err := client.Complete(ctx, &protoReq)
	if err != nil {
		return nil, false, err
	}
	return stream, *protoReq.Stream, nil

}

func RegisterCompletionHandlerClient(ctx context.Context, handler *HttpHandler, client scalellm.CompletionClient) error {

	handler.Handle("POST", "/v1/completions", func(w http.ResponseWriter, req *http.Request) {
		ctx, cancel := context.WithCancel(req.Context())
		defer cancel()
		resp, isStream, err := SendChateRequest(ctx, handler.marshaler, client, req)
		if err != nil {
			DefaultErrorHandler(ctx, handler.marshaler, w, req, err)
			return
		}
		ForwardResponseStream(ctx, handler.marshaler, w, req, isStream, func() (proto.Message, error) { return resp.Recv() })
	})
	return nil
}

func RegisterCompletionHandlerFromEndpoint(ctx context.Context, httpServer *HttpHandler, endpoint string, opts []grpc.DialOption) (err error) {
	conn, err := grpc.DialContext(ctx, endpoint, opts...)
	if err != nil {
		return err
	}
	defer func() {
		if err != nil {
			if cerr := conn.Close(); cerr != nil {
				glog.Errorf("Failed to close conn to %s: %v", endpoint, cerr)
			}
			return
		}
		go func() {
			<-ctx.Done()
			if cerr := conn.Close(); cerr != nil {
				glog.Errorf("Failed to close conn to %s: %v", endpoint, cerr)
			}
		}()
	}()
	return RegisterCompletionHandlerClient(ctx, httpServer, scalellm.NewCompletionClient(conn))
}
