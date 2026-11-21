# src/gpu (owner B). Works as add_subdirectory(src/gpu) from the top level, or standalone for kernel benches:
#   cmake -S src/gpu -B build/gpu -DCMAKE_BUILD_TYPE=Release && cmake --build build/gpu
cmake_minimum_required(VERSION 3.24)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(inkgpu LANGUAGES C CXX OBJCXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_OBJCXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
enable_language(OBJCXX)

file(GLOB INK_METAL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.metal)
string(REPLACE ";" "@@" INK_METAL_LIST "${INK_METAL_SOURCES}")
set(INK_EMBED_CPP ${CMAKE_CURRENT_BINARY_DIR}/embedded_kernels.cpp)
add_custom_command(
  OUTPUT ${INK_EMBED_CPP}
  COMMAND ${CMAKE_COMMAND} -DOUT=${INK_EMBED_CPP} "-DSOURCES=${INK_METAL_LIST}"
          -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_metal.cmake
  DEPENDS ${INK_METAL_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/embed_metal.cmake
  COMMENT "Embedding Metal kernels")

file(GLOB INK_GPU_OPS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/ops*.mm)
add_library(inkgpu STATIC context.mm metal_backend.mm dspark.mm ${INK_GPU_OPS} ${INK_EMBED_CPP})
target_include_directories(inkgpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(inkgpu PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_link_libraries(inkgpu PUBLIC "-framework Metal" "-framework Foundation")
if(NOT TARGET inkbase)
  # Standalone build: compile A's core sources and the CPU reference backend here (for op_diff).
  file(GLOB INK_CORE_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../core/*.cpp)
  add_library(inkbase STATIC ${INK_CORE_SRC})
  target_include_directories(inkbase PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
  set_target_properties(inkbase PROPERTIES CXX_STANDARD 20)
  set(INK_GPU_STANDALONE ON)
endif()
target_link_libraries(inkgpu PUBLIC inkbase)
set_target_properties(inkgpu PROPERTIES CXX_STANDARD 20 OBJCXX_STANDARD 20)

file(GLOB INK_GPU_BENCH CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bench/*.mm)
foreach(src ${INK_GPU_BENCH})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE "-Wl,-force_load,$<TARGET_FILE:inkgpu>" inkgpu)
  target_compile_options(${name} PRIVATE -fobjc-arc)
  set_target_properties(${name} PROPERTIES OBJCXX_STANDARD 20)
endforeach()

# op_diff needs the CPU reference backend (A: src/model/cpu_backend.cpp).
set(INK_CPU_BACKEND ${CMAKE_CURRENT_SOURCE_DIR}/../model/cpu_backend.cpp)
if(EXISTS ${INK_CPU_BACKEND})
  add_executable(op_diff test/op_diff.mm ${INK_CPU_BACKEND})
  target_link_libraries(op_diff PRIVATE "-Wl,-force_load,$<TARGET_FILE:inkgpu>" inkgpu "-framework Accelerate")
  target_compile_options(op_diff PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  set_target_properties(op_diff PROPERTIES OBJCXX_STANDARD 20 CXX_STANDARD 20)
endif()
