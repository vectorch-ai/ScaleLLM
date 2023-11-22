#pragma once
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

namespace llm {
using tcp = boost::asio::ip::tcp;
// a simple http server based on civetweb for serving model metrics and health
// check endpoints
class HttpServer {
 public:
  class Transport;
  using Handler = std::function<bool(Transport&)>;

  HttpServer() = default;

  bool register_uri(const std::string& uri, Handler handler);

  bool start(uint16_t port, int32_t num_threads);

  void stop();

  /**
   * A helper class that request handler can use to query transport related
   * information. one transport object is created for each request, and should
   * be accessed from single thread only.
   */
  class Transport {
   private:
    boost::beast::http::response<boost::beast::http::string_body>* res_;

   public:
    explicit Transport(
        boost::beast::http::response<boost::beast::http::string_body>* res)
        : res_(res) {}

    Transport(const Transport&) = delete;
    Transport& operator=(Transport&) = delete;

    // Send response
    bool send_string(
        const std::string& data,
        const std::string& mime_type = "text/plain; charset=utf-8");

    // Send status code: 200 OK, 503 Service Unavailable, etc.
    bool send_status(int status_code);
  };

 private:
  void async_accept(tcp::acceptor& acceptor);
  void handle_request(tcp::socket& socket);

  // hold the ownership of all request handlers
  std::unordered_map<std::string, Handler> endpoints_;

  // io_context and thread for running the server
  std::unique_ptr<boost::asio::io_context> io_context_;
  std::thread thread_;
};

}  // namespace llm