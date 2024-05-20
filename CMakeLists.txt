cmake_minimum_required(VERSION 3.16.0)

project(cylinder_stitcher_example VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -fPIC -Ofast")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
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
	${CUDA_INCLUDE_DIRS}
	"./build"
)

CUDA_ADD_EXECUTABLE(cylinder_stitcher_example src/core/main.cpp)

SET(HEADER_FILES
	include/stage/driveassistant.h
	include/stage/interactiveimagewidget.h
	include/render/mypainter.h
	include/common/EcalBladeHeightSender.h
	include/stage/TextSourceManager.h
	include/render/Model.h
	include/render/BladeModelTransformer.h
)

SET(SOURCE_FILES
	src/core/FrontStitcherMain.cpp
	src/common/EcalImageSender.cpp
	src/common/GstReceiver.cpp
	src/yolo/yolo.cpp
	src/yolo/yolo_detect.cpp
	src/util/loguru.cpp
	src/common/cylinder_stitcher.cu
	src/render/image_alignment_gpu.cu
	src/calc/nz_block_statistics.cu
	src/calc/x_gn_solver.cu
	src/calc/constraint_terms.cu
	src/calc/reduce.cu
	src/calc/x_pcg_solver.cu
	src/render/project_to_cylinder.cu
	src/render/seam_finder.cu
	src/common/multiband_blend.cu
	src/common/render.cu
	src/stage/driveassistant.cpp
	src/stage/interactiveimagewidget.cpp
	src/render/mypainter.cpp
	src/render/shaders.cpp
	src/render/track.cpp
	src/stage/TextSourceManager.cpp
	src/render/Model.cpp
	src/stage/ThreadInteractiveWidget.cpp
	src/render/BladeModelTransformer.cpp
	src/core/Config.cpp
	src/common/EcalBladeHeightSender.cpp
)

CUDA_ADD_LIBRARY(front_stitcher_main SHARED
	${HEADER_FILES} ${SOURCE_FILES} ${RESOURCES})

set(protobuf_files
	${CMAKE_CURRENT_SOURCE_DIR}/proto/lane_line.proto
	${CMAKE_CURRENT_SOURCE_DIR}/proto/steering_angle.proto
	${CMAKE_CURRENT_SOURCE_DIR}/proto/blade_message.proto
	${CMAKE_CURRENT_SOURCE_DIR}/proto/image.proto
	${CMAKE_CURRENT_SOURCE_DIR}/proto/can.proto
	${CMAKE_CURRENT_SOURCE_DIR}/proto/blade_height.proto
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
	${OPENGL_glu_LIBRARY}
	${OpenCV_LIBS}
	${ASSIMP_LIBRARIES}
	eCAL::core
	glog
	${FREETYPE_LIBRARIES}
)
