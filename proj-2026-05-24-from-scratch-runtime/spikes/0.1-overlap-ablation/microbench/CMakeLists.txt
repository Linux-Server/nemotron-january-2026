cmake_minimum_required(VERSION 3.18)
project(nemotron_microbench LANGUAGES CXX CUDA)

# Point to the libtorch 2.8.0+cu128 C++ SDK (the pinned version — see ../../0.2-pin-and-export/README.md).
#   download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.8.0%2Bcu128.zip
# and: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release ..
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(microbench microbench.cpp)
target_link_libraries(microbench "${TORCH_LIBRARIES}" CUDA::cudart CUDA::nvml)
set_property(TARGET microbench PROPERTY CXX_STANDARD 17)

# ABI note: the _GLIBCXX_USE_CXX11_ABI flag MUST match the libtorch build (cxx11-abi vs pre-cxx11). Mismatch =
# undefined-symbol link errors (the classic libtorch footgun). The cxx11-abi libtorch is the default modern choice.
