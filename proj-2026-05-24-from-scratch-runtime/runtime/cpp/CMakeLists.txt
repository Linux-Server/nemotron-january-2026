cmake_minimum_required(VERSION 3.18)
project(decode LANGUAGES C CXX) # C for vendored picohttpparser; C++ links prebuilt libtorch, no nvcc (glibc-2.41 workaround, as 0.1b)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

# Step 3a: standalone localhost WebSocket tail-latency microbench. Keep this
# target independent from Torch/CUDA so it can be built as a bounded local tool.
add_executable(ws_tail_microbench ws_tail_microbench.cpp)
target_link_libraries(ws_tail_microbench Threads::Threads OpenSSL::Crypto)

add_executable(ws_tail_microbench_client ws_tail_microbench_client.cpp)
target_link_libraries(ws_tail_microbench_client Threads::Threads OpenSSL::Crypto)

if(NOT DEFINED TORCH_ROOT)
  message(STATUS "TORCH_ROOT not set; configured standalone ws_tail targets only")
  return()
endif()
if(NOT DEFINED CUDA_ROOT)
  set(CUDA_ROOT "/usr/local/cuda-12.8")
endif()

set(NEMOTRON_TORCH_INCLUDE_DIRS
  ${TORCH_ROOT}/include
  ${TORCH_ROOT}/include/torch/csrc/api/include
  ${CUDA_ROOT}/targets/x86_64-linux/include)
set(NEMOTRON_TORCH_LINK_DIRS
  ${TORCH_ROOT}/lib
  ${CUDA_ROOT}/lib64
  ${CUDA_ROOT}/lib64/stubs)
set(NEMOTRON_TORCH_LIBS torch torch_cpu torch_cuda c10 c10_cuda cudart pthread)

add_library(nemotron_runtime STATIC
  lib/session/session.cpp
  lib/session/runtime.cpp
  lib/runtime_io/io.cpp
  lib/runtime_io/picohttpparser.c
  lib/admission/density_admission.cpp
  lib/scheduler/batched_steady_scheduler.cpp
  lib/telemetry/stats_collector.cpp
  lib/ws/handshake.cpp
  lib/ws/framing.cpp
  lib/ws/routes.cpp)
target_include_directories(nemotron_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${NEMOTRON_TORCH_INCLUDE_DIRS})
target_compile_definitions(nemotron_runtime PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
target_link_directories(nemotron_runtime PUBLIC ${NEMOTRON_TORCH_LINK_DIRS})
target_link_libraries(nemotron_runtime PUBLIC ${NEMOTRON_TORCH_LIBS} OpenSSL::Crypto)

add_executable(decode decode_main.cpp)
target_include_directories(decode PRIVATE ${TORCH_ROOT}/include ${TORCH_ROOT}/include/torch/csrc/api/include ${CUDA_ROOT}/targets/x86_64-linux/include)
target_compile_definitions(decode PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_link_directories(decode PRIVATE ${TORCH_ROOT}/lib ${CUDA_ROOT}/lib64 ${CUDA_ROOT}/lib64/stubs)
target_link_libraries(decode torch torch_cpu torch_cuda c10 c10_cuda cudart pthread)
set_target_properties(decode PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${CUDA_ROOT}/lib64")

add_executable(pipeline pipeline_main.cpp)
target_include_directories(pipeline PRIVATE ${TORCH_ROOT}/include ${TORCH_ROOT}/include/torch/csrc/api/include ${CUDA_ROOT}/targets/x86_64-linux/include)
target_compile_definitions(pipeline PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_link_directories(pipeline PRIVATE ${TORCH_ROOT}/lib ${CUDA_ROOT}/lib64 ${CUDA_ROOT}/lib64/stubs)
target_link_libraries(pipeline torch torch_cpu torch_cuda c10 c10_cuda cudart pthread)
set_target_properties(pipeline PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${CUDA_ROOT}/lib64")

add_executable(steady steady_main.cpp)
target_include_directories(steady PRIVATE ${TORCH_ROOT}/include ${TORCH_ROOT}/include/torch/csrc/api/include ${CUDA_ROOT}/targets/x86_64-linux/include)
target_compile_definitions(steady PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_link_directories(steady PRIVATE ${TORCH_ROOT}/lib ${CUDA_ROOT}/lib64 ${CUDA_ROOT}/lib64/stubs)
target_link_libraries(steady torch torch_cpu torch_cuda c10 c10_cuda cudart pthread)
set_target_properties(steady PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${CUDA_ROOT}/lib64")

# 1.2b-wire-C++: native AOTIModelPackageLoader harness (closes action D). Build+run in the container (needs the AOTI .so).
add_executable(aoti_encoder aoti_encoder_main.cpp)
target_include_directories(aoti_encoder PRIVATE ${TORCH_ROOT}/include ${TORCH_ROOT}/include/torch/csrc/api/include ${CUDA_ROOT}/targets/x86_64-linux/include)
target_compile_definitions(aoti_encoder PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_link_directories(aoti_encoder PRIVATE ${TORCH_ROOT}/lib ${CUDA_ROOT}/lib64 ${CUDA_ROOT}/lib64/stubs)
target_link_libraries(aoti_encoder torch torch_cpu torch_cuda c10 c10_cuda cudart pthread)
set_target_properties(aoti_encoder PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${CUDA_ROOT}/lib64")

# 1.3b: finalize fork/isolation + decode-continuation harness, with guarded AOTI finalize encoder Phase B.
add_executable(finalize finalize_main.cpp)
target_include_directories(finalize PRIVATE ${TORCH_ROOT}/include ${TORCH_ROOT}/include/torch/csrc/api/include ${CUDA_ROOT}/targets/x86_64-linux/include)
target_compile_definitions(finalize PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_link_directories(finalize PRIVATE ${TORCH_ROOT}/lib ${CUDA_ROOT}/lib64 ${CUDA_ROOT}/lib64/stubs)
target_link_libraries(finalize torch torch_cpu torch_cuda c10 c10_cuda cudart pthread)
set_target_properties(finalize PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${CUDA_ROOT}/lib64")
add_custom_target(aoti_finalize DEPENDS finalize)

# 1.4: single-stream session composition gate, replaying session_bundle.ts through steady AOTI + finalize buckets.
add_executable(session_main session_main.cpp)
target_link_libraries(session_main PRIVATE nemotron_runtime)
set_target_properties(session_main PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${CUDA_ROOT}/lib64")
add_custom_target(session DEPENDS session_main)

# Phase 2 Step 0: native density/concurrency kill-gates over the validated session path.
add_executable(density_main density_main.cpp)
target_link_libraries(density_main PRIVATE nemotron_runtime)
set_target_properties(density_main PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${CUDA_ROOT}/lib64")

add_executable(ws_server ws_server.cpp)
target_link_libraries(ws_server PRIVATE nemotron_runtime Threads::Threads OpenSSL::Crypto)
set_target_properties(ws_server PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${CUDA_ROOT}/lib64")

# STEADY-BATCH-0: fixed-B steady encoder AOTI per-row GPU-time microbench.
add_executable(steady_batch_bench steady_batch_bench.cpp)
target_include_directories(steady_batch_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TORCH_ROOT}/include ${TORCH_ROOT}/include/torch/csrc/api/include ${CUDA_ROOT}/targets/x86_64-linux/include)
target_compile_definitions(steady_batch_bench PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_link_directories(steady_batch_bench PRIVATE ${TORCH_ROOT}/lib ${CUDA_ROOT}/lib64 ${CUDA_ROOT}/lib64/stubs)
target_link_libraries(steady_batch_bench torch torch_cpu torch_cuda c10 c10_cuda cudart pthread)
set_target_properties(steady_batch_bench PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${CUDA_ROOT}/lib64")
