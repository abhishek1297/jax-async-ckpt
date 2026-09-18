cmake_minimum_required(VERSION 3.22)
project(jax_async_ckpt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure dynamic linking RPATH relative to origin ($ORIGIN)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Discover Dependencies
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(URING REQUIRED IMPORTED_TARGET liburing)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE 
  OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

# Core Engine Target
add_library(offloader_engine SHARED src/async_offloader.cpp)
target_include_directories(offloader_engine PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(offloader_engine PUBLIC PkgConfig::URING CUDA::cudart)

# Python Bindings Target
nanobind_add_module(_core_cpp src/bindings.cpp)
target_link_libraries(_core_cpp PRIVATE offloader_engine)

# Install both the extension module and offloader shared library into jax_async_ckpt
install(TARGETS offloader_engine _core_cpp DESTINATION jax_async_ckpt)