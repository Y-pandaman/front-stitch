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

/**
 * 在给定的差异图像中搜索一条最小差异的缝。
 * 
 * @param diff 指向差异图像数据的指针，差异图像用于计算像素间的差异。
 * @param range 搜索范围，限制了在垂直方向上像素可以移动的范围。
 * @param prev_separate_line 前一帧分离线的集合，用于初始化搜索的起始位置。
 * @param separate_line 本次搜索得到的分离线集合，将结果添加到这个集合中。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 * 
 * 函数使用动态编程的方法，从上到下搜索图像中的一条最小差异缝，并将缝的坐标记录在separate_line中。
 * 搜索过程依赖于前一帧的分离线（prev_separate_line）来初始化起始位置，并通过比较当前像素与周围像素的差异值来确定缝的方向。
 */
static inline void search_seam(DIFF_TYPE* diff, int range,
                               std::vector<int> prev_separate_line,
                               std::vector<int>& separate_line, int height,
                               int width) {
    // 初始化搜索的起始位置为前一帧分离线的起始点
    int2 start_loc = make_int2(prev_separate_line[0], 0);
    separate_line.emplace_back(start_loc.x); // 在分离线集合中添加起始点
    int2 loc     = start_loc; // 当前位置
    int last_idx = -1; // 记录上一步选择的方向，用于避免重复搜索

    while (true) {
        // 计算下一步可能到达的3个位置中的最小差异值和方向
        int2 next_loc      = loc + search_seq[0];
        DIFF_TYPE min_diff = diff[next_loc.y * width + next_loc.x];
        int idx            = 0;
        for (int i = 1; i < 3; i++) {
            int2 temp_loc = loc + search_seq[i];

            // 只考虑在搜索范围内且不是上一步选择的方向的像素
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

        loc      = next_loc; // 更新位置
        last_idx = idx; // 更新上一步选择的方向
        if (last_idx == 0)
            separate_line.emplace_back(loc.x); // 如果选择的是正向，则记录该点

        // 如果到达图像的倒数第二行，则记录该点并结束循环
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