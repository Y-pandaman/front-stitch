#ifndef __PROJECT__
#define __PROJECT__

#include "cuda_runtime.h"
#include "cylinder_stitcher.cuh"
#include <cuda.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <vector>

#define CLOSE_ZERO 1.0e-6
// #define STEP 0.002f

// bool projExtraViewToCylinderImage_cuda(ViewGPU extra_view,
//                                        CylinderImageGPU& extra_cyl_image,
//                                        CylinderGPU& cylinder,
//                                        int cyl_image_width,
//                                        int cyl_image_height);

bool proj4ChannelsExtraViewToCylinderImage_cuda(
    ViewGPU4Channels extra_view, CylinderImageGPU4Channels& extra_cyl_image,
    CylinderGPU& cylinder, int cyl_image_width, int cyl_image_height);

bool projToCylinderImage_cuda(std::vector<ViewGPU> views,
                              std::vector<CylinderImageGPU>& cyl_images,
                              CylinderGPU& cylinder, int cyl_image_width,
                              int cyl_image_height);

#endif