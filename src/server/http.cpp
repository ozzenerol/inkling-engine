#include "http.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace ink::http {

std::string status_text(int s) {
  switch (s) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
  }
  return "Status";
}

bool Response::raw(const std::string& s) {
  if (!alive_) return false;
  size_t off = 0;
  while (off < s.size()) {
    ssize_t n = ::send(fd_, s.data() + off, s.size() - off, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      alive_ = false;
      return false;
    }
    off += (size_t)n;
  }
  return true;
}

void Response::send(int status, const std::string& ct, const std::string& body) {
  if (started()) return;
  sent_ = true;
  std::string h = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\nContent-Type: " + ct +
                  "\r\nContent-Length: " + std::to_string(body.size()) +
                  "\r\nAccess-Control-Allow-Origin: *\r\nConnection: keep-alive\r\n\r\n";
  raw(h + body);
}

bool Response::start_stream(int status, const std::string& ct) {
  if (started()) return alive_;
  started_ = streamed_ = true;
  std::string h = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\nContent-Type: " + ct +
                  "\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n"
                  "X-Accel-Buffering: no\r\n\r\n";
  return raw(h);
}

bool Response::write(const std::string& data) { return raw(data); }

bool Response::peer_closed() const {
  if (!alive_) return true;
  pollfd p{fd_, POLLIN, 0};
  if (::poll(&p, 1, 0) <= 0) return false;
  if (p.revents & (POLLHUP | POLLERR)) return true;
  if (p.revents & POLLIN) {
    char c;
    ssize_t n = ::recv(fd_, &c, 1, MSG_PEEK | MSG_DONTWAIT);
    return n == 0;
  }
  return false;
}

namespace {

constexpr size_t kMaxHeader = 64 * 1024;
constexpr size_t kMaxBody = 256ull * 1024 * 1024;

std::string lower(std::string s) {
  for (auto& c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

void connection(int fd, const Handler& h) {
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
  std::string buf;
  char tmp[65536];
  for (;;) {
    size_t hdr_end;
    while ((hdr_end = buf.find("\r\n\r\n")) == std::string::npos) {
      if (buf.size() > kMaxHeader) { ::close(fd); return; }
      ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
      if (n <= 0) { ::close(fd); return; }
      buf.append(tmp, (size_t)n);
    }
    Request req;
    std::string head = buf.substr(0, hdr_end);
    buf.erase(0, hdr_end + 4);
    size_t le = head.find("\r\n");
    std::string line = head.substr(0, le);
    size_t a = line.find(' '), b = line.rfind(' ');
    if (a == std::string::npos || b == a) { ::close(fd); return; }
    req.method = line.substr(0, a);
    std::string target = line.substr(a + 1, b - a - 1);
    size_t q = target.find('?');
    req.path = target.substr(0, q);
    if (q != std::string::npos) req.query = target.substr(q + 1);
    size_t pos = le == std::string::npos ? head.size() : le + 2;
    while (pos < head.size()) {
      size_t e = head.find("\r\n", pos);
      if (e == std::string::npos) e = head.size();
      std::string hl = head.substr(pos, e - pos);
      size_t c = hl.find(':');
      if (c != std::string::npos) {
        std::string v = hl.substr(c + 1);
        size_t s = v.find_first_not_of(" \t");
        req.headers[lower(hl.substr(0, c))] = s == std::string::npos ? "" : v.substr(s);
      }
      pos = e + 2;
    }
    Response res(fd);
    auto te = req.headers.find("transfer-encoding");
    if (te != req.headers.end() && lower(te->second).find("chunked") != std::string::npos) {
      // chunked request body
      for (;;) {
        size_t e;
        while ((e = buf.find("\r\n")) == std::string::npos) {
          ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
          if (n <= 0) { ::close(fd); return; }
          buf.append(tmp, (size_t)n);
        }
        size_t len = std::strtoul(buf.substr(0, e).c_str(), nullptr, 16);
        buf.erase(0, e + 2);
        while (buf.size() < len + 2) {
          ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
          if (n <= 0) { ::close(fd); return; }
          buf.append(tmp, (size_t)n);
        }
        req.body.append(buf, 0, len);
        buf.erase(0, len + 2);
        if (req.body.size() > kMaxBody) { res.send(413, "text/plain", "too large"); ::close(fd); return; }
        if (len == 0) break;
      }
    } else {
      size_t clen = 0;
      auto it = req.headers.find("content-length");
      if (it != req.headers.end()) clen = std::strtoull(it->second.c_str(), nullptr, 10);
      if (clen > kMaxBody) { res.send(413, "text/plain", "too large"); ::close(fd); return; }
      while (buf.size() < clen) {
        ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0) { ::close(fd); return; }
        buf.append(tmp, (size_t)n);
      }
      req.body = buf.substr(0, clen);
      buf.erase(0, clen);
    }
    try {
      h(req, res);
    } catch (const std::exception& e) {
      if (!res.started()) res.send(500, "application/json", std::string("{\"error\":{\"message\":\"internal error\"}}"));
    }
    if (!res.started()) res.send(404, "text/plain", "not found");
    auto conn = req.headers.find("connection");
    bool close_req = conn != req.headers.end() && lower(conn->second) == "close";
    if (!res.keep_alive() || close_req || !res.alive()) { ::close(fd); return; }
  }
}

}  // namespace

void serve(const std::string& host, int port, const Handler& h) {
  int s = ::socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) throw std::runtime_error("socket failed");
  int one = 1;
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)port);
  if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) throw std::runtime_error("bad host " + host);
  if (::bind(s, (sockaddr*)&addr, sizeof addr) < 0)
    throw std::runtime_error("bind " + host + ":" + std::to_string(port) + ": " + std::strerror(errno));
  if (::listen(s, 64) < 0) throw std::runtime_error("listen failed");
  for (;;) {
    int c = ::accept(s, nullptr, nullptr);
    if (c < 0) {
      if (errno == EINTR) continue;
      continue;
    }
    std::thread([c, &h] { connection(c, h); }).detach();
  }
}

}  // namespace ink::http
