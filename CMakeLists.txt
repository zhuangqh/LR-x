cmake_minimum_required(VERSION 3.6)
project(LR_x)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        LR.cpp
        LR.hpp
        main.cpp io.h)

SET( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
IF( NOT EIGEN3_INCLUDE_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )

add_executable(LR_x ${SOURCE_FILES})