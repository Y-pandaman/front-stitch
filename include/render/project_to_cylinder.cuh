#ifndef __PROJECT__
#define __PROJECT__

#include "common/cylinder_stitcher.cuh"
#include "cuda_runtime.h"
#include <cuda.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <vector>


// #define STEP 0.002f

bool proj4ChannelsExtraViewToCylinderImage_cuda(
    ViewGPU4Channels extra_view, CylinderImageGPU4Channels& extra_cyl_image,
    CylinderGPU& cylinder, int cyl_image_width, int cyl_image_height);

bool projToCylinderImage_cuda(std::vector<ViewGPU> views,
                              std::vector<CylinderImageGPU>& cyl_images,
                              CylinderGPU& cylinder, int cyl_image_width,
                              int cyl_image_height);

#endif