#ifndef __BLEND__
#define __BLEND__

#include "common/cylinder_stitcher.cuh"
#include "cuda_runtime.h"
#include <cuda.h>
#include <iostream>
#include <thrust/extrema.h>
#include <vector>

__host__ void MultiBandBlend_cuda(std::vector<CylinderImageGPU> cylImages,
                                  std::vector<uchar*> seam_masks);

#endif