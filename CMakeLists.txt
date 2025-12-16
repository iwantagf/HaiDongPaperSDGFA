# CMakeLists.txt : CMake project for SGDA
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(SGDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "C:/Users/PC/Downloads/libtorch-win-shared-with-deps-2.9.1+cu130/libtorch")

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

add_executable(SGDA "main.cpp" "SGDA.h"  "logistics_regression.h")

target_include_directories(SGDA PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/matplotlib-cpp
)

target_link_libraries(SGDA PRIVATE
    "${TORCH_LIBRARIES}"
    Python3::Python
)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
      "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,\
      $<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,\
      $<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

if (MSVC)
    set(CMAKE_GENERATOR_PLATFORM x64)
    set_property(TARGET SGDA PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()