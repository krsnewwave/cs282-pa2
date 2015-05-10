cmake_minimum_required(VERSION 2.8.4)
project(CS282-PA2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED core imgproc highgui calib3d features2d nonfree)
find_package(OpenGL REQUIRED)


#find_package( PCL 1.2 REQUIRED io common visualization)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

add_library(utils utils.hpp utils.cpp)

set(SOURCE_FILES main.cpp)
add_executable(pa2 ${SOURCE_FILES})
target_link_libraries(pa2 ${OpenCV_LIBS} utils)

add_executable(bonus bonus.cpp)
target_link_libraries(bonus ${OpenCV_LIBS})

add_executable(pa3 main3.cpp)
target_link_libraries(pa3 ${OpenCV_LIBS})
#target_link_libraries(pa3 ${OpenCV_LIBS} ${OPENGL_LIBRARIES})
#target_link_libraries(pa3 ${OpenCV_LIBS} ${PCL_LIBRARIES})
