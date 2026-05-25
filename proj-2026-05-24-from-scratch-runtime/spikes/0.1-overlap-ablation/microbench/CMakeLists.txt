cmake_minimum_required(VERSION 3.18)
project(nemotron_microbench LANGUAGES CXX)   # pure C++; we link prebuilt libtorch .so — NO nvcc needed.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NOTE: we deliberately do NOT use find_package(Torch). TorchConfig.cmake force-`enable_language(CUDA)`, which runs an
# nvcc compiler-probe — and on this box (glibc 2.41) nvcc can't compile even a trivial file (CUDA math_functions.h
# noexcept conflict with bits/mathcalls.h). Since the microbench has no .cu kernels, we just link the prebuilt
# libtorch/CUDA shared libs with g++ and skip nvcc entirely. (A real Wave-2 build with custom kernels WOULD need a
# glibc-compatible CUDA toolkit — flag for the GB10/L40S build envs.)
#
# Configure: cmake -B build \
#   -DTORCH_ROOT="$(<venv>/bin/python -c 'import torch,os;print(os.path.dirname(torch.__file__))')" \
#   -DCUDA_ROOT=/usr/local/cuda-12.8 -DCMAKE_BUILD_TYPE=Release

if(NOT DEFINED TORCH_ROOT)
  message(FATAL_ERROR "pass -DTORCH_ROOT=<site-packages>/torch (the pip torch dir that ships libtorch)")
endif()
if(NOT DEFINED CUDA_ROOT)
  set(CUDA_ROOT "/usr/local/cuda-12.8")
endif()

add_executable(microbench microbench.cpp)
target_include_directories(microbench PRIVATE
  ${TORCH_ROOT}/include
  ${TORCH_ROOT}/include/torch/csrc/api/include
  ${CUDA_ROOT}/targets/x86_64-linux/include)
# ABI MUST match the libtorch build (cxx11_abi=True for this pip torch — verified).
target_compile_definitions(microbench PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_link_directories(microbench PRIVATE ${TORCH_ROOT}/lib ${CUDA_ROOT}/lib64 ${CUDA_ROOT}/lib64/stubs)
target_link_libraries(microbench torch torch_cpu torch_cuda c10 c10_cuda cudart nvidia-ml pthread)
set_target_properties(microbench PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${CUDA_ROOT}/lib64")
