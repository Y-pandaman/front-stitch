cmake_minimum_required(VERSION 3.16.0)

project(stitcher VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -fPIC")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
	set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)

if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# eigen3
include_directories(
	"./include"
	"/usr/include/eigen3/"
	${OpenCV_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	"./src"
	"./"
	"./build"
)

SET(HEADER_FILES
	include/math_utils.h
	include/cylinder_stitcher.h
	include/cylinder_stitcher.cuh
	include/image_alignment_cpu.h
	include/image_alignment_gpu.cuh
	include/nz_block_statistics.cuh
	include/x_gn_solver.cuh
	include/x_pcg_solver.cuh
	include/constraint_terms.cuh
	include/reduce.cuh
	include/innoreal_timer.hpp
	include/project_to_cylinder.cuh
	include/seam_finder.cuh
	include/render.cuh
	include/multiband_blend.cuh
	include/test_JTJ.h
)
SET(SOURCE_FILES
	src/cylinder_stitcher.cpp
	src/cylinder_stitcher.cu
	src/image_alignment_cpu.cpp
	src/image_alignment_gpu.cu
	src/nz_block_statistics.cu
	src/x_gn_solver.cu
	src/constraint_terms.cu
	src/reduce.cu
	src/x_pcg_solver.cu
	src/project_to_cylinder.cu
	src/seam_finder.cu
	src/multiband_blend.cu
	src/render.cu
	src/test_JTJ.cpp
)

CUDA_ADD_LIBRARY(stitcher SHARED ${HEADER_FILES} ${SOURCE_FILES} ${RESOURCES})

# opencv
target_link_libraries(stitcher ${CUDA_cusparse_LIBRARY})

target_link_libraries(
	stitcher
	${OpenCV_LIBS}
	${CUDA_LIBRARIES}
	${CUDA_cusparse_LIBRARY}
)