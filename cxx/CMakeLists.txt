cmake_minimum_required(VERSION 3.13)
project(XPBDRods LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)                       # specify C++ standard
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_BUILD_TYPE Debug)

include_directories("./")
include_directories("LBFGSpp/include/")

find_package(Eigen3 3.3 REQUIRED)   # find Eigen

# add_library(CosseratRod Cosserat.cpp)
# target_link_libraries(CosseratRod Eigen3::Eigen)


set(ALGLIB_SRC_DIR "../alglib-cpp/src")
set(ALGLIB_SOURCES
    ${ALGLIB_SRC_DIR}/ap.cpp
    ${ALGLIB_SRC_DIR}/alglibinternal.cpp
    ${ALGLIB_SRC_DIR}/linalg.cpp
    ${ALGLIB_SRC_DIR}/alglibmisc.cpp
    ${ALGLIB_SRC_DIR}/solvers.cpp
    ${ALGLIB_SRC_DIR}/optimization.cpp
)

add_library(AlglibOptimization)
target_sources(AlglibOptimization
    PRIVATE
        ${ALGLIB_SOURCES}
)

add_executable(TestEnergy tests/test_energy.cpp)
target_link_libraries(TestEnergy Eigen3::Eigen)

add_executable(TestSE3Jacobian tests/test_se3_jacobian.cpp)
target_link_libraries(TestSE3Jacobian Eigen3::Eigen)

add_executable(TestTipPosition tests/test_tip_position.cpp)
target_link_libraries(TestTipPosition Eigen3::Eigen)

add_executable(TestEnergyGradient tests/test_energy_gradient.cpp)
target_link_libraries(TestEnergyGradient Eigen3::Eigen)

add_executable(TestOptimization tests/test_optimization.cpp)
target_link_libraries(TestOptimization Eigen3::Eigen)

add_executable(TestWriteToFile tests/test_write_to_file.cpp)
target_link_libraries(TestWriteToFile Eigen3::Eigen)

add_executable(TestPeristalticRobot tests/test_peristaltic_robot.cpp)
target_link_libraries(TestPeristalticRobot Eigen3::Eigen AlglibOptimization)

add_executable(TestAlglib tests/test_alglib.cpp)
target_link_libraries(TestAlglib Eigen3::Eigen AlglibOptimization)

add_executable(TestPeristalticRobotMotion tests/test_peristaltic_robot_motion.cpp)
target_link_libraries(TestPeristalticRobotMotion Eigen3::Eigen AlglibOptimization)