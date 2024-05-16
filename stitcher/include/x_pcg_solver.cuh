#ifndef GALAHOLOSYSTEM_X_PCG_SOLVER_H
#define GALAHOLOSYSTEM_X_PCG_SOLVER_H

#include <cusparse_v2.h>
#include <thrust/device_vector.h>

class DotMuliplyWrapper;
class SparseMvWrapper;
class InnerProductWrapper;
class InnerProductWrapper;
class AxpyOperatorWrapper;
class UpdateZAndRWrapper;
class InnerProductWrapper;
class XpayOperatorWrapper;

class PcgLinearSolverCPUGPU {
public:
    PcgLinearSolverCPUGPU(int row);
    ~PcgLinearSolverCPUGPU();

    void Solve(float* d_x, int* d_ia, int* d_ja, float* d_a, int num_nnz,
               float* d_b, float* precond, int max_iter);

private:
    int row_              = 0;
    const float floatone_ = 1.0f, m_floatzero = 0.0f;
    float alpha_ = 0.0f, beta_ = 0.0f, zr_dot_old_ = 0.0f, zr_dot_ = 0.0f;

    thrust::device_vector<float> d_buf_;
    std::vector<float> buf_;
    float *d_p_, *d_omega_;
    float *omega_, *p_, *r_, *z_, *delta_x_, *precond_;

    SparseMvWrapper* spmv_wrapper_;
    cusparseHandle_t cusparseHandle_;
    cusparseMatDescr_t descr_;
};

class PcgLinearSolverGPU {
public:
    PcgLinearSolverGPU(int row);
    ~PcgLinearSolverGPU();

    void Solve(float* d_x, int* d_ia, int* d_ja, float* d_a, int num_nnz,
               float* d_b, float* precond, int max_iter);

private:
    int row_              = 0;
    const float floatone_ = 1.0f, m_floatzero = 0.0f;
    float alpha_ = 0.0f, beta_ = 0.0f, zr_dot_old_ = 0.0f, zr_dot_ = 0.0f;

    thrust::device_vector<float> d_buf_;
    float *d_omega_, *d_p_, *d_r_, *d_z_, *d_delta_x_;

    thrust::device_vector<float> d_zr_dot_, d_zr_dot_old_, d_alpha_, d_beta_;

    DotMuliplyWrapper* dm_wrapper_;
    SparseMvWrapper* spmv_wrapper_;
    InnerProductWrapper* ip_wrapper_;
    InnerProductWrapper* ip_wrapper_alpha_;
    AxpyOperatorWrapper* ao_wrapper_;
    UpdateZAndRWrapper* uzar_wrapper_;
    InnerProductWrapper* ip_wrapper_beta_;
    XpayOperatorWrapper* xo_wrapper_;

    cusparseHandle_t cusparseHandle_;
    cusparseMatDescr_t descr_;
};

#endif   // GALAHOLOSYSTEM_X_PCG_SOLVER_H
