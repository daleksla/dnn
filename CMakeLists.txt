cmake_minimum_required(VERSION 3.1)
PROJECT(dnn)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_BUILD_TYPE RelWithDebInfo)

cmake_policy(SET CMP0072 NEW) # to use GLVND (Nvidia-vendor OpenGL impl)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

list(PREPEND CMAKE_MODULE_PATH
    ${PROJECT_SOURCE_DIR}/cmake/Modules)

find_package( OpenCV 4.5.0 REQUIRED )

message( ${CMAKE_MODULE_PATH} )
message( ${OpenCV_INCLUDE_DIRS} )

add_library( dnn ${CMAKE_CURRENT_SOURCE_DIR}/src/dnn.cpp )
target_include_directories( dnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} )
TARGET_LINK_LIBRARIES( dnn PRIVATE ${OpenCV_LIBS} )

add_executable( dnn_runner ${CMAKE_CURRENT_SOURCE_DIR}/src/runner.cpp )
target_link_libraries( dnn_runner dnn )

#add_library( easydnn ${CMAKE_CURRENT_SOURCE_DIR}/src/easydnn.cpp )
#target_include_directories( easydnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} )
#TARGET_LINK_LIBRARIES( easydnn PRIVATE ${OpenCV_LIBS} )
#
#add_executable( easydnn_runner ${CMAKE_CURRENT_SOURCE_DIR}/src/runner.cpp )
#target_link_libraries( easydnn_runner easydnn )
