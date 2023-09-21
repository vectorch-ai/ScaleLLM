package main

import (
	"context"
	"io"
	"net/http"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"google.golang.org/genproto/googleapis/api/httpbody"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// ForwardResponseStream forwards the stream from gRPC server to REST client.
func ForwardResponseStream(ctx context.Context, marshaler runtime.Marshaler, w http.ResponseWriter, req *http.Request, recv func() (proto.Message, error)) {
	f, ok := w.(http.Flusher)
	if !ok {
		grpclog.Infof("Flush not supported in %T", w)
		http.Error(w, "unexpected type of web server", http.StatusInternalServerError)
		return
	}

	// write the server metadata to the response
	w.Header().Set("Transfer-Encoding", "chunked")

	var delimiter = []byte("\n")

	var wroteHeader bool
	var buf []byte
	for {
		resp, err := recv()
		if err == io.EOF {
			if err := writeData(w, []byte("[DONE]\n")); err != nil {
				grpclog.Infof("Failed to send delimiter chunk: %v", err)
			}
			return
		}
		if err != nil {
			handleForwardResponseStreamError(ctx, wroteHeader, marshaler, w, req, err, delimiter)
			return
		}

		if !wroteHeader {
			w.Header().Set("Content-Type", "text/event-stream")
		}
		httpBody, isHTTPBody := resp.(*httpbody.HttpBody)
		switch {
		case resp == nil:
			buf, err = marshaler.Marshal(errorChunk(status.New(codes.Internal, "empty response")))
		case isHTTPBody:
			buf = httpBody.GetData()
		default:
			buf, err = marshaler.Marshal(resp)
		}

		if err != nil {
			grpclog.Infof("Failed to marshal response chunk: %v", err)
			handleForwardResponseStreamError(ctx, wroteHeader, marshaler, w, req, err, delimiter)
			return
		}
		buf = append(buf, delimiter...)
		if err := writeData(w, buf); err != nil {
			grpclog.Infof("Failed to send response chunk: %v", err)
			return
		}
		wroteHeader = true
		f.Flush()
	}
}

func writeData(w http.ResponseWriter, data []byte) error {
	// write data
	if _, err := w.Write([]byte("data: ")); err != nil {
		grpclog.Infof("Failed to send response chunk: %v", err)
		return err
	}
	if _, err := w.Write(data); err != nil {
		grpclog.Infof("Failed to send response chunk: %v", err)
		return err
	}
	return nil
}

func handleForwardResponseStreamError(ctx context.Context, wroteHeader bool, marshaler runtime.Marshaler, w http.ResponseWriter, req *http.Request, err error, delimiter []byte) {
	st := status.Convert(err)
	msg := errorChunk(st)

	if !wroteHeader {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(runtime.HTTPStatusFromCode(st.Code()))
	}
	buf, err := marshaler.Marshal(msg)
	if err != nil {
		grpclog.Infof("Failed to marshal an error: %v", err)
		return
	}
	if _, err := w.Write(buf); err != nil {
		grpclog.Infof("Failed to notify error to client: %v", err)
		return
	}
	if _, err := w.Write(delimiter); err != nil {
		grpclog.Infof("Failed to send delimiter chunk: %v", err)
		return
	}
}

func errorChunk(st *status.Status) map[string]proto.Message {
	return map[string]proto.Message{"error": st.Proto()}
}
