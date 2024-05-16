#include "seam_finder.cuh"

__global__ void color_diff_kernel(CylinderImageGPU left, CylinderImageGPU mid,
                                  CylinderImageGPU right, DIFF_TYPE* diff,
                                  int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        uchar3 left_rgb  = left.getImageValue(x, y);
        uchar3 mid_rgb   = mid.getImageValue(x, y);
        uchar3 right_rgb = right.getImageValue(x, y);

        uchar left_mask  = left.getMaskValue(x, y);
        uchar mid_mask   = mid.getMaskValue(x, y);
        uchar right_mask = right.getMaskValue(x, y);

        float3 l = make_float3((float)left_rgb.x, (float)left_rgb.y,
                               (float)left_rgb.z) /
                   MASK_MAX;
        float3 m =
            make_float3((float)mid_rgb.x, (float)mid_rgb.y, (float)mid_rgb.z) /
            MASK_MAX;
        float3 r = make_float3((float)right_rgb.x, (float)right_rgb.y,
                               (float)right_rgb.z) /
                   MASK_MAX;

        float diff_lm = 0.05f, diff_mr = 0.05;
        if (left_mask == 255 && mid_mask == 255)
            diff_lm = fabs(l.x - m.x) + fabs(l.y - m.y) + fabs(l.z - m.z);
        if (right_mask == 255 && mid_mask == 255)
            diff_mr = fabs(r.x - m.x) + fabs(r.y - m.y) + fabs(r.z - m.z);
        diff[pixelIdx] = (diff_lm + diff_mr) * DIFF_TYPE_MAX_VALUE;

        pixelIdx += total_thread;
    }
}

__global__ void mask_op_kernel(CylinderImageGPU left, CylinderImageGPU mid,
                               CylinderImageGPU right, uchar* total_mask,
                               uchar* left_only_mask, uchar* right_only_mask,
                               uchar* overlap_mask, int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        uchar l = left.getMaskValue(x, y) / MASK_MAX;
        uchar m = mid.getMaskValue(x, y) / MASK_MAX;
        uchar r = right.getMaskValue(x, y) / MASK_MAX;

        uchar total   = l | m | r;
        uchar l_only  = (1 - l * m) * l;
        uchar r_only  = (1 - m * r) * r;
        l_only        = l_only - (l_only * r_only);
        uchar overlap = l * m * r;

        total_mask[pixelIdx]      = total * MASK_MAX;
        left_only_mask[pixelIdx]  = l_only;
        right_only_mask[pixelIdx] = r_only;
        overlap_mask[pixelIdx]    = overlap;
        pixelIdx += total_thread;
    }
}

__global__ void get_seam_mask(CylinderImageGPU left, CylinderImageGPU mid,
                              CylinderImageGPU right,
                              const uchar* left_mask_only,
                              const uchar* total_mask,
                              const uchar* right_mask_only,
                              const int* separate_line, uchar* seam_mask_left,
                              uchar* seam_mask_mid, uchar* seam_mask_right,
                              int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
        int x  = pixelIdx % width;
        int y  = pixelIdx / width;
        int x1 = separate_line[y];
        int x2 = separate_line[y + height];

        if (left_mask_only[pixelIdx]) {
            seam_mask_left[pixelIdx] = MASK_MAX;
        } else if (right_mask_only[pixelIdx]) {
            seam_mask_right[pixelIdx] = MASK_MAX;
        } else {
            if (x < x1)
                seam_mask_left[pixelIdx] = left.getMaskValue(x, y);
            else if (x < x2)
                seam_mask_mid[pixelIdx] = mid.getMaskValue(x, y);
            else
                seam_mask_right[pixelIdx] = right.getMaskValue(x, y);
        }

        seam_mask_mid[pixelIdx] = total_mask[pixelIdx] -
                                  seam_mask_left[pixelIdx] -
                                  seam_mask_right[pixelIdx];

        pixelIdx += total_thread;
    }
}

__host__ bool SeamFind_cuda(std::vector<CylinderImageGPU> cylImages, int height,
                            int width,
                            std::vector<std::vector<int>>& separate_line,
                            int range, std::vector<uchar*>& seam_masks,
                            bool is_first) {
    assert(cylImages.size() == 3);

    int numPixel   = height * width;
    int num_thread = 512;
    int num_block  = min(65535, (numPixel + num_thread - 1) / num_thread);

    uchar *_left_mask_only, *_total_mask, *_right_mask_only, *_overlap_mask;
    cudaMalloc((void**)&_left_mask_only, sizeof(uchar) * numPixel);
    cudaMalloc((void**)&_total_mask, sizeof(uchar) * numPixel);
    cudaMalloc((void**)&_right_mask_only, sizeof(uchar) * numPixel);
    cudaMalloc((void**)&_overlap_mask, sizeof(uchar) * numPixel);

    DIFF_TYPE* _diff;
    cudaMalloc((void**)&_diff, sizeof(DIFF_TYPE) * numPixel);
    DIFF_TYPE* diff;
    cudaHostAlloc((void**)&diff, sizeof(DIFF_TYPE) * numPixel,
                  cudaHostAllocDefault);

    int* _separate_line;
    cudaMalloc((void**)&_separate_line, sizeof(int) * height * 2);

    mask_op_kernel<<<num_block, num_thread>>>(
        cylImages[0], cylImages[1], cylImages[2], _total_mask, _left_mask_only,
        _right_mask_only, _overlap_mask, height, width);

    color_diff_kernel<<<num_block, num_thread>>>(
        cylImages[0], cylImages[1], cylImages[2], _diff, height, width);

    cudaMemcpy(diff, _diff, sizeof(DIFF_TYPE) * numPixel,
               cudaMemcpyDeviceToHost);

    std::vector<std::vector<int>> new_separate_line;
    new_separate_line.resize(2);

    search_seam(diff, range, separate_line[0], new_separate_line[0], height,
                width);
    search_seam(diff, range, separate_line[1], new_separate_line[1], height,
                width);

    if (is_first == false) {
        for (int i = 0; i < separate_line[0].size(); ++i) {
            new_separate_line[0][i] = 0.5 * (float)separate_line[0][i] +
                                      0.5 * (float)new_separate_line[0][i];
            new_separate_line[1][i] = 0.5 * (float)separate_line[1][i] +
                                      0.5 * (float)new_separate_line[1][i];
        }
    }

    cudaMemcpy(_separate_line, new_separate_line[0].data(),
               sizeof(int) * height, cudaMemcpyHostToDevice);
    cudaMemcpy(_separate_line + height, new_separate_line[1].data(),
               sizeof(int) * height, cudaMemcpyHostToDevice);

    separate_line = new_separate_line;

    get_seam_mask<<<num_block, num_thread>>>(
        cylImages[0], cylImages[1], cylImages[2], _left_mask_only, _total_mask,
        _right_mask_only, _separate_line, seam_masks[0], seam_masks[1],
        seam_masks[2], height, width);

    cudaFree(_left_mask_only);
    cudaFree(_total_mask);
    cudaFree(_right_mask_only);
    cudaFree(_overlap_mask);
    cudaFree(_diff);
    cudaFreeHost(diff);
    cudaFree(_separate_line);

    return true;
}
