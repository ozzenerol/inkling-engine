# Inkling engine (top level, owner A).
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cmake_minimum_required(VERSION 3.24)
project(inkling LANGUAGES C CXX OBJCXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_OBJCXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Core utilities (A): json, safetensors, config, thread pool.
add_library(inkbase STATIC src/core/json.cpp src/core/safetensors.cpp src/core/config.cpp src/core/parallel.cpp)
target_include_directories(inkbase PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Metal backend + kernels (B).
add_subdirectory(src/gpu)

# Model + engine (A; speculative.cpp B).
file(GLOB INK_ENGINE_SRC CONFIGURE_DEPENDS src/engine/*.cpp)
add_library(inkcore STATIC src/model/cpu_backend.cpp src/model/model.cpp ${INK_ENGINE_SRC})
target_include_directories(inkcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(inkcore PUBLIC inkbase "-Wl,-force_load,$<TARGET_FILE:inkgpu>" inkgpu "-framework Accelerate")

# CLI and reference test tool (A).
add_executable(inktest tests/ref/inktest.cpp)
target_link_libraries(inktest PRIVATE inkcore)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/inkling.cpp)
  add_executable(inkling tools/inkling.cpp)
  target_link_libraries(inkling PRIVATE inkcore)
endif()

# Tokenizer + server (C). After inkcore: the server links the real engine if(TARGET inkcore).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer/CMakeLists.txt)
  add_subdirectory(src/tokenizer)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/server/CMakeLists.txt)
  add_subdirectory(src/server)
endif()
