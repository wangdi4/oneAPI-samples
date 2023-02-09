if(UNIX)
    # Direct CMake to use icpx rather than the default C++ compiler/linker
    set(CMAKE_CXX_COMPILER icpx)
else() # Windows
    # Force CMake to use icx-cl rather than the default C++ compiler/linker 
    # (needed on Windows only)
    set(CMAKE_CXX_COMPILER icx)
endif()

cmake_minimum_required(VERSION 3.23.0)

project(ACFixed CXX)

find_package(IntelSYCL REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory (src)
