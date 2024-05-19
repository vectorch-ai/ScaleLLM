#include "http_server.h"

#include <glog/logging.h>

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <memory>
#include <string>
#include <utility>

namespace llm {

bool HttpServer::register_uri(const std::string& uri,
                              HttpServer::Handler handler) {
  if (endpoints_.count(uri) != 0) {
    return false;
  }
  endpoints_[uri] = handler;
  return true;
}

void HttpServer::async_accept(tcp::acceptor& acceptor) {
  // create a new socket
  const auto socket = std::make_shared<tcp::socket>(*io_context_);
  acceptor.async_accept(
      *socket, [this, &acceptor, socket](boost::system::error_code ec) {
        // loop to accept new incoming connections
        async_accept(acceptor);

        if (!ec) {
          try {
            // handle the request
            handle_request(*socket);
          } catch (const std::exception& e) {
            LOG(ERROR) << "Exception in processing request: " << e.what();
          }
        } else {
          LOG(ERROR) << "Error in accepting connection: " << ec.message();
        }
      });
}

void HttpServer::handle_request(tcp::socket& socket) {
  // process the request
  boost::beast::flat_buffer buffer;
  boost::beast::http::request<boost::beast::http::string_body> req;
  boost::beast::http::read(socket, buffer, req);
  boost::beast::http::response<boost::beast::http::string_body> res;
  res.version(req.version());
  res.keep_alive(req.keep_alive());
  const std::string target = req.target();
  auto it = endpoints_.find(target);
  if (it == endpoints_.end()) {
    res.result(boost::beast::http::status::not_found);
    res.body() = "The resource '" + target + "' was not found.";
    res.set(boost::beast::http::field::content_type, "text/plain");
  } else {
    auto& handler = it->second;
    Transport transport(&res);
    if (!handler(transport)) {
      res.result(boost::beast::http::status::internal_server_error);
      res.body() = "An error occurred processing the request.";
      res.set(boost::beast::http::field::content_type, "text/plain");
    }
  }
  boost::beast::http::write(socket, res);
  socket.shutdown(tcp::socket::shutdown_send);
}

bool HttpServer::start(uint16_t port, int32_t num_threads) {
  io_context_ = std::make_unique<boost::asio::io_context>(num_threads);

  thread_ = std::thread([this, port] {
    tcp::acceptor acceptor{*io_context_, tcp::endpoint{tcp::v4(), port}};
    async_accept(acceptor);
    io_context_->run();
  });
  LOG(INFO) << "Started http server on 0.0.0.0:" << port;
  return true;
}

void HttpServer::stop() {
  // notifiy io_context to stop
  if (io_context_) {
    io_context_->stop();
  }
  // wait for thread to finish
  if (thread_.joinable()) {
    thread_.join();
  }
  endpoints_.clear();
}

bool HttpServer::Transport::send_string(const std::string& data,
                                        const std::string& mime_type) {
  res_->set(boost::beast::http::field::content_type, mime_type);
  res_->body() = data;
  res_->result(boost::beast::http::status::ok);
  return true;
}

bool HttpServer::Transport::send_status(int status_code) {
  res_->result(status_code);
  return true;
}

}  // namespace llm