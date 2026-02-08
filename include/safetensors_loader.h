//
// Created by rahim on 08/02/26.
//

#ifndef SAFETENSORS_LOADER_CPP_SAFETENSORS_LOADER_H
#define SAFETENSORS_LOADER_CPP_SAFETENSORS_LOADER_H

#pragma once

#include "mmap_file.h"
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <vector>

// ---------------- dtype mapping ----------------
inline torch::ScalarType to_torch_dtype(const std::string& dtype) {
    if (dtype == "F32") return torch::kFloat32;
    if (dtype == "F16") return torch::kFloat16;
    if (dtype == "BF16") return torch::kBFloat16;
    if (dtype == "I64") return torch::kInt64;
    throw std::runtime_error("Unsupported dtype: " + dtype);
}

inline size_t dtype_size(const std::string& dtype) {
    if (dtype == "F32") return 4;
    if (dtype == "F16") return 2;
    if (dtype == "BF16") return 2;
    if (dtype == "I64") return 8;
    throw std::runtime_error("Unsupported dtype: " + dtype);
}

// ---------------- loader ----------------
class SafeTensorLoader {
public:
    explicit SafeTensorLoader(const std::string& path)
        : file_(path) {
        parse_header();
    }

    std::unordered_map<std::string, torch::Tensor>
    load(torch::Device device, bool zero_copy = true) {

        std::unordered_map<std::string, torch::Tensor> result;

        for (const auto& [name, meta] : header_.items()) {

            // ✅ REQUIRED BY SAFETENSORS SPEC
            if (name == "__metadata__")
                continue;

            // Extra safety
            if (!meta.contains("shape") ||
                !meta.contains("dtype") ||
                !meta.contains("data_offsets")) {
                continue;
            }

            const auto shape =
                meta["shape"].get<std::vector<int64_t>>();

            const auto offsets =
                meta["data_offsets"].get<std::vector<size_t>>();

            const auto dtype =
                meta["dtype"].get<std::string>();

            if (offsets.size() != 2)
                throw std::runtime_error(
                    "Invalid data_offsets for tensor: " + name
                );

            // ---- validate size ----
            size_t expected_elems = 1;
            for (auto s : shape) expected_elems *= s;

            size_t expected_bytes =
                expected_elems * dtype_size(dtype);

            if (offsets[1] - offsets[0] != expected_bytes) {
                throw std::runtime_error(
                    "Size mismatch for tensor: " + name
                );
            }

            void* tensor_data =
                static_cast<void*>(data_start_ + offsets[0]);

            auto options = torch::TensorOptions()
                .dtype(to_torch_dtype(dtype))
                .device(torch::kCPU);

            auto t = torch::from_blob(
                tensor_data,
                shape,
                options
            );

            if (!zero_copy)
                t = t.clone();

            result.emplace(name, t.to(device));
        }

        return result;
    }

private:
    void parse_header() {
        auto* base =
            static_cast<uint8_t*>(file_.data());

        if (file_.size() < 8)
            throw std::runtime_error("File too small");

        uint64_t header_size =
            *reinterpret_cast<uint64_t*>(base);

        uint8_t* header_start = base + 8;

        if (8 + header_size > file_.size())
            throw std::runtime_error("Invalid header size");

        data_start_ = header_start + header_size;

        header_ = nlohmann::json::parse(
            std::string(
                reinterpret_cast<char*>(header_start),
                header_size
            )
        );
    }
    MMapFile file_;
    nlohmann::json header_;
    uint8_t* data_start_{nullptr};
};

#endif // SAFETENSORS_LOADER_CPP_SAFETENSORS_LOADER_H