// Owner: A. Safetensors index: parses headers of all shards in a model dir (no data read).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "core/tensor.h"

namespace ink {

class SafeTensors {
 public:
  struct Entry {
    std::string name;
    std::string file;      // absolute path
    DType dtype;
    int ndim = 0;
    int64_t shape[4] = {0, 0, 0, 0};
    uint64_t offset = 0;   // absolute byte offset in file
    uint64_t bytes = 0;
  };
  // Adds every *.safetensors file in dir (or the single file `path`). Names may be prefixed.
  void add_file(const std::string& path, const std::string& prefix = "");
  void add_dir(const std::string& dir, const std::string& exclude_substr = "mtp.safetensors");
  const Entry* find(const std::string& name) const;
  const Entry& at(const std::string& name) const;  // throws with the name
  const std::map<std::string, Entry>& all() const { return entries_; }
  uint64_t total_bytes() const;

 private:
  std::map<std::string, Entry> entries_;
};

// Read raw bytes of an entry into dst (pread; for tests and the CPU backend).
void read_entry(const SafeTensors::Entry& e, void* dst);

}  // namespace ink
