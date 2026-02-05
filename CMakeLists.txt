cmake_minimum_required(VERSION 3.20)

project(SmartiVisionTask)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "C:/smarti/opencv/build/x64/vc16/lib")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SmartiVisionTask src/main.cpp)

target_link_libraries(SmartiVisionTask ${OpenCV_LIBS})