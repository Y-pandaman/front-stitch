#ifndef GALAHOLOSYSTEM_GN_SOLVER_H
#define GALAHOLOSYSTEM_GN_SOLVER_H

#include "constraint_terms.cuh"
#include <opencv2/opencv.hpp>
#include <thrust/device_vector.h>

struct SolverPara {
    float data_term_weight_;
    bool has_data_term_;
    float smooth_term_weight_;
    bool has_smooth_term_;
    float zero_term_weight_;
    bool has_zero_term_;

    SolverPara(float data_term_weight, bool has_data_term,
               float smooth_term_weight, bool has_smooth_term,
               float zero_term_weight, bool has_zero_term) {
        data_term_weight_   = data_term_weight;
        has_data_term_      = has_data_term;
        smooth_term_weight_ = smooth_term_weight;
        has_smooth_term_    = has_smooth_term;
        zero_term_weight_   = zero_term_weight;
        has_zero_term_      = has_zero_term;
    }

    SolverPara() { }
};

class SparseMatrixCSR {
public:
    SparseMatrixCSR() : nnz_(0), row_(0), col_(0) { }

    SparseMatrixCSR(int row, int col) : nnz_(0) {
        row_ = row;
        col_ = col;
    }

    int row_;
    int col_;
    int nnz_;

    thrust::device_vector<int> d_ia_;    // row_ptr, size: row_ + 1
    thrust::device_vector<int> d_ja_;    // col_ind, size: nnz_
    thrust::device_vector<float> d_a_;   // val, size: nnz_
};

class NzBlockStatisticsForJTJ;
class PcgLinearSolverCPUGPU;
class PcgLinearSolverGPU;

class GNSolver {
public:
    GNSolver(int img_rows, int img_cols, int node_img_rows, int node_img_cols,
             int* d_pixel_rela_idx_vec, float* d_pixel_rela_weight_vec,
             int* d_node_rela_idx_vec, float* d_node_rela_weight_vec,
             float2* d_original_node_vec, float2* d_node_vec,
             cv::cuda::GpuMat& d_src_img, cv::cuda::GpuMat& d_target_img,
             cv::cuda::GpuMat& d_dy_img, cv::cuda::GpuMat& d_dx_img,
             float triangle_u, float triangle_v, const SolverPara& para);
    ~GNSolver();

public:
    static char* Stype() {
        return "GN_Solver";
    }

    bool Next();
    void UpdateNodes();
    void CalcWarpedPixels();

    cv::cuda::GpuMat &d_src_img_, &d_target_img_;
    cv::cuda::GpuMat &d_dy_img_, &d_dx_img_;
    float2 *d_original_node_vec_, *d_node_vec_;
    float triangle_u_, triangle_v_;

    int img_rows_, img_cols_, node_img_rows_, node_img_cols_;
    int node_num_, pixel_num_, triangle_num_;
    int* d_pixel_rela_idx_vec_;
    float* d_pixel_rela_weight_vec_;
    int* d_node_rela_idx_vec_;
    float* d_node_rela_weight_vec_;

    std::vector<Constraint*> cons_;
    SparseMatrixCSR* d_JTJ_;
    thrust::device_vector<float> d_JTb_;
    thrust::device_vector<float> d_preconditioner_;
    thrust::device_vector<float> d_delta_;
    thrust::device_vector<float2> d_warped_pixel_vec_;
    int vars_num_;

    int counter_iter_;
    float residual_;

    SolverPara para_;
    NzBlockStatisticsForJTJ* nz_blocks_static_;
    PcgLinearSolverGPU* pcg_linear_solver_;
    void InitJTJ();
};

#endif   // GALAHOLOSYSTEM_GN_SOLVER_H
