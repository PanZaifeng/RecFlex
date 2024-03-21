cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(custom_schedules LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)

file(GLOB SOURCES
    *.cc
    *.cu
)

include_directories(include)
add_library(schedules SHARED ${SOURCES})

find_package(Python REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import os; import RecFlex; print(os.path.dirname(RecFlex.__file__))" OUTPUT_VARIABLE RECFLEX_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(SCHED_BASIC_DIR "${RECFLEX_DIR}/sched_basic_csrc")
include_directories("${SCHED_BASIC_DIR}/include")
target_link_libraries(schedules "${SCHED_BASIC_DIR}/build/libsched_basic.so")
