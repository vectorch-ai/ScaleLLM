#pragma once
#include <absl/strings/numbers.h>
#include <evhtp/evhtp.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

struct mg_connection;
namespace llm {

// a simple http server based on civetweb for serving model metrics and health
// check endpoints
class HttpServer {
 public:
  class Transport;
  using Handler = std::function<bool(Transport&)>;

  HttpServer() = default;

  bool RegisterURI(const std::string& uri, Handler handler);

  bool Start(uint16_t port, int32_t num_threads);

  void Stop();

  /**
   * A helper class that request handler can use to query transport related
   * information. one transport object is created for each request, and should
   * be accessed from single thread only.
   */
  class Transport {
   private:
    evhtp_request_t* req_;

   public:
    explicit Transport(evhtp_request_t* req) : req_(req) {}

    Transport(const Transport&) = delete;
    Transport& operator=(Transport&) = delete;

    // Get request
    htp_method GetMethod() const;

    std::optional<std::string> GetParam(const std::string& name) const;

    template <typename int_type>
    std::optional<int_type> GetIntParam(const std::string& name) const {
      auto str_val = GetParam(name);
      if (!str_val) {
        return false;
      }
      int_type value;
      if (absl::SimpleAtoi(str_val.value(), value)) {
        return value;
      }
      return std::nullopt;
    }

    // Send response
    bool SendString(const std::string& data,
                    const std::string& mime_type = "text/plain; charset=utf-8");
  };

 private:
  // http request callback to dispatch to registered handlers
  static void Dispatch(evhtp_request_t* req, void* arg);

  // event callback to stop the event loop
  static void StopCallback(evutil_socket_t fd, short what, void* arg);

  // hold the ownership of all request handlers
  std::unordered_map<std::string, Handler> endpoints_;

  evhtp_t* htp_;
  event_base* evbase_;
  evutil_socket_t fds_[2];
  event* break_ev_;

  std::thread thread_;
};

}  // namespace llm