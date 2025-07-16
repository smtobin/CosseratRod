cmake_minimum_required(VERSION 3.13)
project(XPBDRods LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)                       # specify C++ standard
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_BUILD_TYPE Debug)

include_directories("./")

find_package(Eigen3 3.3 REQUIRED)   # find Eigen

add_library(CosseratRod Cosserat.cpp)
target_link_libraries(CosseratRod Eigen3::Eigen)

add_executable(TestEnergy tests/test_energy.cpp)
target_link_libraries(TestEnergy CosseratRod Eigen3::Eigen)

add_executable(TestSE3Jacobian tests/test_se3_jacobian.cpp)
target_link_libraries(TestSE3Jacobian Eigen3::Eigen)

add_executable(TestTipPosition tests/test_tip_position.cpp)
target_link_libraries(TestTipPosition CosseratRod Eigen3::Eigen)

add_executable(TestEnergyGradient tests/test_energy_gradient.cpp)
target_link_libraries(TestEnergyGradient CosseratRod Eigen3::Eigen)