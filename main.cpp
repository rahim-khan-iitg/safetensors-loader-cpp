#include <iostream>
#include "safetensors_loader.h"


int main() {
  std::cout << "hello\n";

  SafeTensorLoader loader(
      "/home/rahim/xelp/work/SmolLM2-135M-Instruct/model.safetensors"
  );

  auto weights = loader.load(
      torch::kCPU,
      true   // zero-copy
  );

  std::cout << "Loaded " << weights.size() << " tensors\n";

  for (const auto& [name, tensor] : weights) {
    std::cout << name << " -> " << tensor.sizes() << "\n";
    std::cout<<tensor<<std::endl;
    break;
  }

  return 0;
}