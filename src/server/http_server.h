#pragma once
#include <CivetServer.h>
#include <absl/strings/numbers.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

struct mg_connection;
namespace llm {

// a simple http server based on civetweb for serving model metrics and health
// check endpoints
class HttpServer {
 public:
  class Transport;
  using Handler = std::function<bool(std::unique_ptr<Transport>)>;

  HttpServer() = default;

  bool RegisterURI(const std::string& uri, Handler handler);

  void Start(uint16_t port, uint32_t num_threads = 2);

  void Stop();

  /**
   * A helper class that request handler can use to query transport related
   * information. one transport object is created for each request, and should
   * be accessed from single thread only.
   */
  class Transport {
   private:
    mg_connection* conn_;

   public:
    explicit Transport(mg_connection* conn) : conn_(conn) {}

    Transport(const Transport&) = delete;
    Transport& operator=(Transport&) = delete;

    // Get request
    std::string GetMethod() const;

    bool GetParam(const std::string& name, std::string* value) const;

    template <typename int_type>
    bool GetIntParam(const std::string& name, int_type* value) const {
      std::string str_val;
      if (!GetParam(name, &str_val)) {
        return false;
      }
      return absl::SimpleAtoi(str_val, value);
    }

    // Send response
    bool SendString(const std::string& data,
                    const std::string& mime_type = "text/plain; charset=utf-8");
  };

 private:
  std::unique_ptr<CivetServer> server_;

  // hold the ownership of all request handlers
  std::unordered_map<std::string, std::unique_ptr<CivetHandler>> endpoints_;
};

}  // namespace llm