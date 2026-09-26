cmake_minimum_required(VERSION 3.20)
project(stitch_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT OpenCV_DIR)
  set(OpenCV_DIR "C:/opencv/build")
endif()

# The installed OpenCV Windows pack is the vc16 (VS2019) world DLL.
# Newer MSVC can link that import library; the stock OpenCVConfig.cmake
# does not map VS2026 onto vc16 unless the runtime is set explicitly.
if(NOT OpenCV_ARCH)
  set(OpenCV_ARCH x64)
endif()
if(NOT OpenCV_RUNTIME)
  set(OpenCV_RUNTIME vc16)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d)

add_library(stitch_core SHARED src/stitch_core.cpp)
target_include_directories(stitch_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(stitch_core PRIVATE ${OpenCV_LIBS})

if(MSVC)
  target_compile_options(stitch_core PRIVATE /utf-8 /EHsc /W3)
  target_compile_definitions(stitch_core PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
