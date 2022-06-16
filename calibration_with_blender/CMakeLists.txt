cmake_minimum_required(VERSION 3.16)

project(calibration_with_blender VERSION 0.1.0)

#if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
#    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
#    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
#            "${CMAKE_BINARY_DIR}/conan.cmake"
#            TLS_VERIFY ON)
#endif()
#
#include(${CMAKE_BINARY_DIR}/conan.cmake)
#
#conan_cmake_configure(REQUIRES opencv/4.5.5
#        GENERATORS cmake_find_package)
#
#conan_cmake_autodetect(settings)
#
#conan_cmake_install(PATH_OR_REFERENCE .
#        BUILD missing
#        REMOTE conancenter
#        SETTINGS ${settings})

find_package(OpenCV REQUIRED)

add_executable(image_distort image_distort.cpp)
add_executable(calibration_benchmark calibration_benchmark.cpp)

target_link_libraries(image_distort PRIVATE opencv_highgui opencv_calib3d opencv_imgproc)
target_link_libraries(calibration_benchmark PRIVATE opencv_highgui opencv_calib3d opencv_imgproc)
