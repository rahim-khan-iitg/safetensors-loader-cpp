//
// Created by rahim on 08/02/26.
//

#ifndef SAFETENSORS_LOADER_CPP_MAP_FILE_H
#define SAFETENSORS_LOADER_CPP_MAP_FILE_H
// mmap_file.h
#pragma once
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>

class MMapFile {
public:
  explicit MMapFile(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
      throw std::runtime_error("Failed to open file");

    size_ = lseek(fd_, 0, SEEK_END);
    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);

    if (data_ == MAP_FAILED)
      throw std::runtime_error("mmap failed");
  }

  ~MMapFile() {
    if (data_ && data_ != MAP_FAILED)
      munmap(data_, size_);
    if (fd_ >= 0)
      close(fd_);
  }

  void* data() const { return data_; }
  size_t size() const { return size_; }

private:
  int fd_{-1};
  void* data_{nullptr};
  size_t size_{0};
};
#endif // SAFETENSORS_LOADER_CPP_MAP_FILE_H
