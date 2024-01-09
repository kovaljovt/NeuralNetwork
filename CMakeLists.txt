cmake_minimum_required(VERSION 3.27)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 REQUIRED)

# Add include directories for Eigen
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(NeuralNetwork
        main.cpp
        src/NeuralNetwork.cpp
)

target_link_libraries(NeuralNetwork PRIVATE Eigen3::Eigen)
