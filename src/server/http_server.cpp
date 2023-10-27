#include "http_server.h"

#include <CivetServer.h>
#include <civetweb.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llm {

class GetRequestHandler : public CivetHandler {
 private:
  HttpServer::Handler handler_;

 public:
  explicit GetRequestHandler(HttpServer::Handler handler)
      : handler_(std::move(handler)) {}

  bool handleGet(CivetServer*, struct mg_connection* conn) override {
    return handler_(std::make_unique<HttpServer::Transport>(conn));
  }
};

bool HttpServer::RegisterURI(const std::string& uri,
                             HttpServer::Handler handler) {
  if (endpoints_.count(uri) != 0) {
    return false;
  }
  endpoints_[uri] = std::make_unique<GetRequestHandler>(std::move(handler));
  return true;
}

void HttpServer::Start(uint16_t port, uint32_t num_threads) {
  // clang-format off
  const std::vector<std::string> options = {"listening_ports", std::to_string(port),
                                            "num_threads", std::to_string(num_threads),
                                            "enable_keep_alive", "yes",
                                            "request_timeout_ms", "60000",
                                            "document_root", "."};
  // clang-format on
  server_ = std::make_unique<CivetServer>(options);
  // register uri handlers
  for (const auto& [uri, handler] : endpoints_) {
    server_->addHandler(uri, handler.get());
  }
}

void HttpServer::Stop() {
  if (server_) {
    server_->close();
    server_.reset();
  }
  endpoints_.clear();
}

std::string HttpServer::Transport::GetMethod() const {
  return CivetServer::getMethod(conn_);
}

bool HttpServer::Transport::GetParam(const std::string& name,
                                     std::string* value) const {
  return CivetServer::getParam(conn_, name.c_str(), *value);
}

bool HttpServer::Transport::SendString(const std::string& data,
                                       const std::string& mime_type) {
  /* Send HTTP message header */
  mg_send_http_ok(conn_, mime_type.c_str(), data.size());
  /* Send HTTP message content */
  mg_write(conn_, data.data(), data.size());
  return true;
}

}  // namespace llm