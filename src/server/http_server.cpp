#include "http_server.h"

#include <event2/util.h>
#include <evhtp/evhtp.h>
#include <glog/logging.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llm {

bool HttpServer::RegisterURI(const std::string& uri,
                             HttpServer::Handler handler) {
  if (endpoints_.count(uri) != 0) {
    return false;
  }
  endpoints_[uri] = handler;
  return true;
}

void HttpServer::Dispatch(evhtp_request_t* req, void* arg) {
  HttpServer* self = static_cast<HttpServer*>(arg);
  const std::string uri = req->uri->path->full;
  auto it = self->endpoints_.find(uri);
  if (it == self->endpoints_.end()) {
    evhtp_send_reply(req, EVHTP_RES_NOTFOUND);
    return;
  }
  Transport transport(req);
  if (!it->second(transport)) {
    evhtp_send_reply(req, EVHTP_RES_SERVERR);
  }
}

void HttpServer::StopCallback(evutil_socket_t /*fd*/,
                              short /*what*/,
                              void* arg) {
  event_base_loopbreak(static_cast<event_base*>(arg));
}

void HttpServer::Start(uint16_t port, int32_t num_threads) {
  evbase_ = event_base_new();
  htp_ = evhtp_new(evbase_, nullptr);
  evhtp_enable_flag(htp_, EVHTP_FLAG_ENABLE_NODELAY);
  // set request handler callback
  evhtp_set_gencb(htp_, Dispatch, this);
  // set thread number
  evhtp_use_threads_wexit(htp_, nullptr, nullptr, num_threads, nullptr);
  if (evhtp_bind_socket(htp_, "0.0.0.0", port, 1024) != 0) {
    LOG(FATAL) << "Failed to bind to port " << port;
  }

  // set up a pipe to break the event loop
  if (evutil_socketpair(AF_UNIX, SOCK_STREAM, 0, fds_) == -1) {
    LOG(FATAL) << "Failed to create socket pair";
  }
  break_ev_ = event_new(evbase_, fds_[0], EV_READ, StopCallback, evbase_);
  event_add(break_ev_, nullptr);
  thread_ = std::thread(event_base_loop, evbase_, 0);
}

void HttpServer::Stop() {
  if (thread_.joinable()) {
    // notify the event loop to stop
    send(fds_[1], &evbase_, sizeof(event_base*), 0);
    // wait for the event loop to stop
    thread_.join();

    event_free(break_ev_);
    evutil_closesocket(fds_[0]);
    evutil_closesocket(fds_[1]);
    evhtp_unbind_socket(htp_);
    evhtp_free(htp_);
    event_base_free(evbase_);
  }
  endpoints_.clear();
}

htp_method HttpServer::Transport::GetMethod() const { return req_->method; }

bool HttpServer::Transport::GetParam(const std::string& name,
                                     std::string* value) const {
  // return CivetServer::getParam(conn_, name.c_str(), *value);
  evhtp_uri_t* uri = req_->uri;
  evhtp_kv_t* kv = nullptr;
  TAILQ_FOREACH(kv, uri->query, next) {
    if (std::string(kv->key, kv->klen) == name) {
      *value = std::string(kv->val, kv->vlen);
      return true;
    }
  }
  return false;
}

bool HttpServer::Transport::SendString(const std::string& data,
                                       const std::string& mime_type) {
  // Send HTTP message header
  evhtp_headers_add_header(
      req_->headers_out,
      evhtp_header_new("Content-Type", mime_type.c_str(), 1, 1));
  // Send HTTP message content
  evbuffer_add(req_->buffer_out, data.data(), data.size());
  // Send HTTP message status
  evhtp_send_reply(req_, EVHTP_RES_OK);
  return true;
}

}  // namespace llm