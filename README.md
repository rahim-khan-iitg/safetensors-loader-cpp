# Safetensors Loader C++

A high-performance C++ header-only loader for [Safetensors](https://github.com/huggingface/safetensors) format, designed for seamless integration with LibTorch.

## Features

- **C++20**: modern C++ implementation.
- **Zero-Copy**: uses `mmap` for fast loading and low memory overhead.
- **LibTorch Integrated**: returns tensors directly as `torch::Tensor`.
- **Lightweight**: minimal dependencies (nlohmann_json and LibTorch).
- **Supports Multiple Dtypes**: F32, F16, BF16, and I64.

## Prerequisites

- **CMake**: version 3.15 or higher.
- **Conan**: for dependency management (nlohmann_json).
- **LibTorch**: PyTorch C++ distribution.
- **CUDA**: (Optional) if running on GPU.

## Build Instructions

### 1. Install Dependencies
This project uses Conan to manage `nlohmann_json`.

```bash
conan install . --output-folder=conan_deps --build=missing
```

### 2. Configure and Build
Ensure you have the path to your LibTorch installation.

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_TOOLCHAIN_FILE=conan_deps/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

## Usage Example

```cpp
#include "safetensors_loader.h"
#include <iostream>

int main() {
    // Initialize the loader
    SafeTensorLoader loader("path/to/model.safetensors");

    // Load tensors to specific device (e.g., CPU)
    // Second argument: true for zero-copy (from_blob), false for clone
    auto weights = loader.load(torch::kCPU, true);

    std::cout << "Loaded " << weights.size() << " tensors" << std::endl;

    // Iterate through tensors
    for (const auto& [name, tensor] : weights) {
        std::cout << "Tensor: " << name << " | Shape: " << tensor.sizes() << std::endl;
        // Break after first one for demo
        break;
    }

    return 0;
}
```

## License

This project is licensed under the MIT License.
