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

	// importing generated stubs
	scalellm "gateway/proto"

	gw "github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
)

func SendModelsRequest(ctx context.Context, marshaler gw.Marshaler, client scalellm.ModelsClient, req *http.Request) (*scalellm.ListResponse, error) {
	// TODO: add support for path parameters
	var protoReq scalellm.ListRequest
	newReader, berr := utilities.IOReaderFactory(req.Body)
	if berr != nil {
		return nil, status.Errorf(codes.InvalidArgument, "%v", berr)
	}
	if err := marshaler.NewDecoder(newReader()).Decode(&protoReq); err != nil && err != io.EOF {
		return nil, status.Errorf(codes.InvalidArgument, "%v", err)
	}

	resp, err := client.List(ctx, &protoReq)
	if err != nil {
		return nil, err
	}
	return resp, nil

}

func RegisterModelsHandlerClient(ctx context.Context, handler *HttpHandler, client scalellm.ModelsClient) error {

	handler.Handle("GET", "/v1/models", func(w http.ResponseWriter, req *http.Request) {
		ctx, cancel := context.WithCancel(req.Context())
		defer cancel()
		resp, err := SendModelsRequest(ctx, handler.marshaler, client, req)
		if err != nil {
			DefaultErrorHandler(ctx, handler.marshaler, w, req, err)
			return
		}
		ForwardResponseMessage(ctx, handler.marshaler, w, req, resp)
	})
	return nil
}

func RegisterModelsHandlerFromEndpoint(ctx context.Context, httpServer *HttpHandler, endpoint string, opts []grpc.DialOption) (err error) {
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
	return RegisterModelsHandlerClient(ctx, httpServer, scalellm.NewModelsClient(conn))
}
