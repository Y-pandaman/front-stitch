#ifndef __SEAM_FINDER__
#define __SEAM_FINDER__

#include "cuda_runtime.h"
#include "cylinder_stitcher.cuh"
#include "math_utils.h"
#include "omp.h"
#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <time.h>

#define MASK_MAX 255
#define DIFF_TYPE ushort
#define DIFF_TYPE_MAX_VALUE 65535
#define TEST_TIME false

const int2 search_seq[3] = {make_int2(0, 1), make_int2(1, 0), make_int2(-1, 0)};

static inline void search_seam(DIFF_TYPE* diff, int range,
                               std::vector<int> prev_separate_line,
                               std::vector<int>& separate_line, int height,
                               int width) {
    int2 start_loc = make_int2(prev_separate_line[0], 0);
    separate_line.emplace_back(start_loc.x);
    int2 loc     = start_loc;
    int last_idx = -1;
    while (true) {
        int2 next_loc      = loc + search_seq[0];
        DIFF_TYPE min_diff = diff[next_loc.y * width + next_loc.x];
        int idx            = 0;
        for (int i = 1; i < 3; i++) {
            int2 temp_loc = loc + search_seq[i];

            if (temp_loc.x >= prev_separate_line[temp_loc.y] - range &&
                temp_loc.x < prev_separate_line[temp_loc.y] + range &&
                last_idx != i) {
                DIFF_TYPE d = diff[temp_loc.y * width + temp_loc.x];
                if (d < min_diff) {
                    min_diff = d;
                    idx      = 3 - i;
                    next_loc = temp_loc;
                }
            }
        }

        loc      = next_loc;
        last_idx = idx;
        if (last_idx == 0)
            separate_line.emplace_back(loc.x);
        if (loc.y == height - 2) {
            separate_line.emplace_back(loc.x);
            break;
        }
    }
}

__host__ bool SeamFind_cuda(std::vector<CylinderImageGPU> cylImages, int height,
                            int width,
                            std::vector<std::vector<int>>& separate_line,
                            int range, std::vector<uchar*>& seam_masks,
                            bool is_first);

#endif