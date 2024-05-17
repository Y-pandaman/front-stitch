cmake_minimum_required(VERSION 3.16.0)

project(cylinder_stitcher_example VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -fPIC -Ofast")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Qt5 COMPONENTS Widgets OpenGL Gui Core REQUIRED)
find_package(OpenMP)
find_package(Freetype REQUIRED)
find_package(eCAL REQUIRED)
find_package(Protobuf REQUIRED)
find_package(assimp REQUIRED)

if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
	include/
	"/usr/include/eigen3/"
	${FREETYPE_INCLUDE_DIRS}
	${eCAL_INCLUDE_DIRS}
	"../stitcher"
	"../ar_vis"
	"./build"
)

CUDA_ADD_EXECUTABLE(cylinder_stitcher_example src/main.cpp)

CUDA_ADD_LIBRARY(front_stitcher_main SHARED
	include/FrontStitcherMain.h src/FrontStitcherMain.cpp
	src/EcalImageSender.cpp include/EcalImageSender.h src/GstReceiver.cpp
	src/yolo.cpp src/yolo_detect.cpp src/loguru.cpp)

set(protobuf_files
	${CMAKE_CURRENT_SOURCE_DIR}/proto/lane_line.proto
	${CMAKE_CURRENT_SOURCE_DIR}/proto/image.proto
)

protobuf_target_cpp(front_stitcher_main proto ${protobuf_files})

target_link_libraries(
	cylinder_stitcher_example
	front_stitcher_main
)

target_link_libraries(
	front_stitcher_main
	Qt5::Widgets Qt5::OpenGL Qt5::Gui Qt5::Core
	${CUDA_LIBRARIES}
	${CUDA_cusparse_LIBRARY}
	stitcher
	ar_vis
	${OPENGL_glu_LIBRARY}
	${OpenCV_LIBS}
	${ASSIMP_LIBRARIES}
	eCAL::core
	glog
	${FREETYPE_LIBRARIES}
)