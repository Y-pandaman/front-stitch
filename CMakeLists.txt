# requirement: opencv-4.5.2, Qt5.9.9, eigen3, cmake_minimum_required(VERSION 3.5)
cmake_minimum_required(VERSION 3.16.0)

project(cylinder_stitcher VERSION 1.0.0 LANGUAGES CXX)

include_directories(./include)
ADD_SUBDIRECTORY(./stitcher)
ADD_SUBDIRECTORY(./ar_vis)
ADD_SUBDIRECTORY(./example)