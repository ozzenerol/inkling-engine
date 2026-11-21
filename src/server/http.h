// Minimal blocking HTTP/1.1 server: one thread per connection, keep-alive, Content-Length bodies, and
// streaming responses (SSE) that end by closing the connection. Owner: C.
#pragma once
#include <functional>
#include <map>
#include <string>

namespace ink::http {

struct Request {
  std::string method, path, query;
  std::map<std::string, std::string> headers;  // lower-case names
  std::string body;
};

class Response {
 public:
  explicit Response(int fd) : fd_(fd) {}
  // Whole response at once (keeps the connection alive).
  void send(int status, const std::string& content_type, const std::string& body);
  // Start a stream: headers, then any number of write() calls; the connection is closed afterwards.
  bool start_stream(int status, const std::string& content_type);
  bool write(const std::string& data);  // false once the client is gone
  bool alive() const { return alive_; }
  bool started() const { return started_ || sent_; }
  bool keep_alive() const { return sent_ && !streamed_; }
  // Non-blocking check whether the peer closed the connection.
  bool peer_closed() const;

 private:
  bool raw(const std::string& s);
  int fd_;
  bool alive_ = true, started_ = false, sent_ = false, streamed_ = false;
};

using Handler = std::function<void(const Request&, Response&)>;

// Blocks forever. Throws std::runtime_error if it cannot bind.
void serve(const std::string& host, int port, const Handler& h);

std::string status_text(int status);

}  // namespace ink::http
