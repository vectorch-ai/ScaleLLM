package main

import (
	"context"
	"io"
	"net/http"
	"strings"

	"github.com/golang/glog"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	gw "github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
)

// A HandlerFunc handles a specific pair of path pattern and HTTP method.
type HandlerFunc func(w http.ResponseWriter, r *http.Request)

// ServeMux is a request multiplexer for grpc-gateway.
// It matches http requests to patterns and invokes the corresponding handler.
type HttpHandler struct {
	// handlers maps HTTP method to a list of handlers.
	handlers  map[string][]handler
	marshaler gw.Marshaler
}

// NewServeMux returns a new ServeMux whose internal mapping is empty.
func NewHttpHandler(marshaler gw.Marshaler) *HttpHandler {
	server := &HttpHandler{
		handlers:  make(map[string][]handler),
		marshaler: marshaler,
	}
	return server
}

// Handle associates "h" to the pair of HTTP method and path pattern.
func (s *HttpHandler) Handle(meth string, path string, h HandlerFunc) {
	s.handlers[meth] = append([]handler{{path: path, h: h}}, s.handlers[meth]...)
}

// ServeHTTP dispatches the request to the first handler whose pattern matches to r.Method and r.URL.Path.
func (s *HttpHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	path := r.URL.Path
	if !strings.HasPrefix(path, "/") {
		DefaultRoutingErrorHandler(ctx, s.marshaler, w, r, http.StatusBadRequest)
		return
	}
	path = strings.TrimSuffix(path, "/")

	for _, h := range s.handlers[r.Method] {
		if h.path == path && h.h != nil {
			h.h(w, r)
			return
		}
	}
	DefaultRoutingErrorHandler(ctx, s.marshaler, w, r, http.StatusNotFound)
}

type handler struct {
	path string
	h    HandlerFunc
}

func DefaultRoutingErrorHandler(ctx context.Context, marshaler gw.Marshaler, w http.ResponseWriter, r *http.Request, httpStatus int) {
	sterr := status.Error(codes.Internal, "Unexpected routing error")
	switch httpStatus {
	case http.StatusBadRequest:
		sterr = status.Error(codes.InvalidArgument, http.StatusText(httpStatus))
	case http.StatusMethodNotAllowed:
		sterr = status.Error(codes.Unimplemented, http.StatusText(httpStatus))
	case http.StatusNotFound:
		sterr = status.Error(codes.NotFound, http.StatusText(httpStatus))
	}
	DefaultErrorHandler(ctx, marshaler, w, r, sterr)
}

func DefaultErrorHandler(ctx context.Context, marshaler gw.Marshaler, w http.ResponseWriter, r *http.Request, err error) {
	// return Internal when Marshal failed
	const fallback = `{"error": {"code": 13, "message": "failed to marshal error message"}}`
	s := status.Convert(err)
	pb := errorChunk(s)

	w.Header().Set("Content-Type", "application/json")

	if s.Code() == codes.Unauthenticated {
		w.Header().Set("WWW-Authenticate", s.Message())
	}

	buf, merr := marshaler.Marshal(pb)
	if merr != nil {
		glog.Errorf("Failed to marshal error message %q: %v", s, merr)
		w.WriteHeader(http.StatusInternalServerError)
		if _, err := io.WriteString(w, fallback); err != nil {
			glog.Errorf("Failed to write response: %v", err)
		}
		return
	}

	st := gw.HTTPStatusFromCode(s.Code())
	w.WriteHeader(st)
	if _, err := w.Write(buf); err != nil {
		glog.Errorf("Failed to write response: %v", err)
		return
	}
	// write delimiter
	if _, err := w.Write([]byte("\n\n")); err != nil {
		glog.Errorf("Failed to send delimiter chunk: %v", err)
	}
}
