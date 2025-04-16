cmake_minimum_required(VERSION 3.10)
project(pcl_learn_workspace)

cmake_policy(SET CMP0144 NEW) # 添加此行以消除FLANN_ROOT警告

find_package(PCL REQUIRED)
find_package(OpenMP)

include_directories(
    ${PCL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 主程序
add_executable(main
    main.cpp
)

target_link_libraries(main
    ${PCL_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found, enabling OpenMP support.")
    target_link_libraries(main OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found, compiling without OpenMP support.")
endif()

#设置调试模式
set(CMAKE_BUILD_TYPE "Release")
#针对平台的编译优化
target_compile_options(main PRIVATE -march=native -mtune=native)