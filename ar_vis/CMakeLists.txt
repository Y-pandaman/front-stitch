cmake_minimum_required(VERSION 3.1.0)

project(ar_vis)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -fPIC")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(CMAKE_VERSION VERSION_LESS "3.7.0")
        set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED PATHS C:/Program Files/opencv/opencv-4.5.2/x64/vc14/lib/)
find_package(Qt5 COMPONENTS Widgets OpenGL Gui Core REQUIRED PATHS C:/Qt/Qt5.9.9/5.9.9/msvc2015_64/lib/cmake/Qt5/)
find_package(OpenMP)
find_package(assimp REQUIRED)
find_package(Freetype REQUIRED)
find_package(CUDA REQUIRED)
find_package(eCAL REQUIRED)

if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
        ./include
        "/usr/include/eigen3/"
        ${eCAL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
)

SET(HEADER_FILES
        include/driveassistant.h
        include/interactiveimagewidget.h
        include/mypainter.h
        include/EcalBladeHeightSender.h
        include/TextSourceManager.h
        include/Model.h
        include/BladeModelTransformer.h
)
SET(SOURCE_FILES
        src/driveassistant.cpp
        src/interactiveimagewidget.cpp
        src/mypainter.cpp
        src/shaders.cpp
        src/track.cpp
        src/TextSourceManager.cpp
        src/Model.cpp
        src/ThreadInteractiveWidget.cpp
        src/BladeModelTransformer.cpp
        src/EcalImageSender.cpp
        src/Config.cpp
        src/EcalBladeHeightSender.cpp
)

CUDA_ADD_LIBRARY(ar_vis SHARED ${HEADER_FILES} ${SOURCE_FILES} ${RESOURCES}
)

set(protobuf_files
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/lane_line.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/steering_angle.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/blade_message.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/image.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/can.proto
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/blade_height.proto
)

protobuf_target_cpp(ar_vis proto ${protobuf_files})

# link
target_link_libraries(
        ar_vis
        Qt5::Widgets Qt5::OpenGL Qt5::Gui Qt5::Core
        ${OPENGL_glu_LIBRARY}
        ${OpenCV_LIBS}
        ${CUDA_cusparse_LIBRARY}
)
