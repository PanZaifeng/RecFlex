cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(recom_sched_basic)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)

file(GLOB SOURCES
    *.cc
    *.cu
)

include_directories(include)
add_library(sched_basic SHARED ${SOURCES})

target_compile_options(sched_basic PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        -std=c++17;
                        -Xcompiler;
                        -D_GLIBCXX_USE_CXX11_ABI=1;
                       >)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python_INCLUDE_DIRS})
target_link_libraries(sched_basic ${Python_LIBRARIES})

execute_process(COMMAND "${Python_EXECUTABLE}" -c "import os; import torch; print(os.path.dirname(torch.__file__))" OUTPUT_VARIABLE TORCH_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
target_link_libraries(sched_basic ${TORCH_LIBRARIES})
