cmake_minimum_required(VERSION 3.15)

# Default to Conan toolchain if not specified
if(NOT CMAKE_TOOLCHAIN_FILE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/conan_deps/conan_toolchain.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/conan_deps/conan_toolchain.cmake" CACHE FILEPATH "Conan toolchain")
endif()

# Default to Release build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

project(safetensors_cpp CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(LIBTORCH_ROOT "/home/rahim/xelp/work/bnb-cpp/libtorch/libtorch")
list(PREPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
#set(SRC_FILES
#        src/environment_vars.cpp
#        src/yolo_model.cpp
#)

find_package(Torch REQUIRED)
find_package(nlohmann_json REQUIRED)

#add_executable(safetensors_cpp main.cpp ${SRC_FILES})
add_executable(safetensors_cpp main.cpp)

target_include_directories(safetensors_cpp
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Link dependencies
find_library(CUPTI_LIB NAMES libcupti-fccdc5fc.so.12 PATHS "${LIBTORCH_ROOT}/lib" NO_DEFAULT_PATH)

target_link_libraries(safetensors_cpp
        nlohmann_json::nlohmann_json
        ${TORCH_LIBRARIES}
        ${CUPTI_LIB}
)
