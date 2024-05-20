#ifndef GALAHOLOSYSTEM_RELA_STATISTICS_H
#define GALAHOLOSYSTEM_RELA_STATISTICS_H

#include <thrust/device_vector.h>

struct NzBlockStatisticsForJTJ {
    int nz_block_num_;
    thrust::device_vector<int> d_nz_block_idx_vec_;
    thrust::device_vector<int> d_function_idx_vec_;
    thrust::device_vector<int> d_pixel_idx_num_vec_;
    thrust::device_vector<int> d_offset_vec_;

    thrust::device_vector<int> d_pre_nz_block_num_vec_;
    thrust::device_vector<int> d_row_offset_vec_;

    void clear() {
        d_function_idx_vec_.clear();
        d_offset_vec_.clear();
        d_pixel_idx_num_vec_.clear();
        d_nz_block_idx_vec_.clear();

        d_pre_nz_block_num_vec_.clear();
        d_row_offset_vec_.clear();
    }

    void CalcNzBlocksAndFunctions(int pixel_num, int triangle_num, int node_num,
                                  int* d_pixel_rela_idx_vec,
                                  float* d_pixel_rela_weight_vec,
                                  int* d_node_rela_idx_vec,
                                  float* d_node_rela_weight_vec);
};

#endif   // GALAHOLOSYSTEM_RELA_STATISTICS_H
