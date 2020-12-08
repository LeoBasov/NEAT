cmake_minimum_required(VERSION 3.10)

project(neat_xor_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
file(GLOB_RECURSE sources ../../src/*.cc *.cc)
list(FILTER sources EXCLUDE REGEX "../src/main.cc")
add_executable(xor_test ${sources})
target_link_libraries(xor_test ${EIGEN3_LIBRARIES} OpenMP::OpenMP_CXX)
