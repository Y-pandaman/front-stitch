#include "cuda_utils.h"
#include "helper_cuda.h"
#include "x_pcg_solver.cuh"
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>

struct DotMuliplyWrapper {
    int num_data;
    const float* input_data1;
    const float* input_data2;
    float* output;
    __device__ void operator()() {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < num_data) {
            output[idx] = input_data1[idx] * input_data2[idx];
        }
    }
};
__global__ void DotMuliplyKernel(DotMuliplyWrapper dm_wrapper) {
    dm_wrapper();
}

struct InnerProductWrapper {
    int num_data;
    const float* input_data1;
    const float* input_data2;
    float* output;
    __device__ void operator()() {
        __shared__ float s_res[32];
        int warp_id = threadIdx.x / 32;
        if (threadIdx.x - warp_id * 32 == 31) {
            s_res[warp_id] = 0;
        }

        float value = 0.f;
        for (int iter = threadIdx.x; iter < num_data; iter += blockDim.x) {
            value += input_data1[iter] * input_data2[iter];
        }
        for (int i = 16; i >= 1; i /= 2) {
            value += __shfl_xor_sync(__activemask(), value, i, 32);
        }
        if (threadIdx.x - warp_id * 32 == 31) {
            s_res[warp_id] = value;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float res_out = 0.f;
            for (int iter = 0; iter < 32; ++iter) {
                res_out += s_res[iter];
            }
            output[0] = res_out;
        }
    }
};
__global__ void InnerProductKernel(InnerProductWrapper ip_wrapper) {
    ip_wrapper();
}

__global__ void UpdateAlphaKernel(float* d_alpha, float* d_zr_dot) {
    if (threadIdx.x == 0) {
        d_alpha[0] = d_zr_dot[0] / d_alpha[0];
    }
}

struct AxpyOperatorWrapper {
    int num_data;
    const float* scale_a;
    const float* vector_x;
    float* vector_y;

    __device__ void operator()() {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < num_data) {
            vector_y[idx] += scale_a[0] * vector_x[idx];
        }
    }
};
__global__ void AxpyOperatorKernel(AxpyOperatorWrapper ao_wrapper) {
    ao_wrapper();
}

struct UpdateZAndRWrapper {
    int num_data;
    const float* omega;
    const float* preconditioner;
    const float* scale_a;
    float* d_r;
    float* d_z;

    __device__ void operator()() {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < num_data) {
            float res_r = d_r[idx] - scale_a[0] * omega[idx];
            d_r[idx]    = res_r;
            d_z[idx]    = preconditioner[idx] * res_r;
        }
    }
};
__global__ void UpdateZAndRKernel(UpdateZAndRWrapper uzar_wrapper) {
    uzar_wrapper();
}

__global__ void UpdataBetaKernel(float* d_zr_dot, float* d_zr_dot_old,
                                 float* d_beta) {
    if (threadIdx.x == 0) {
        d_beta[0] = d_zr_dot[0] / d_zr_dot_old[0];
    }
}

struct XpayOperatorWrapper {
    int num_data;
    const float* scale_a;
    const float* vector_x;
    float* vector_y;

    __device__ void operator()() {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < num_data) {
            vector_y[idx] = scale_a[0] * vector_y[idx] + vector_x[idx];
        }
    }
};
__global__ void XapyOperatorKernel(XpayOperatorWrapper xo_wrapper) {
    xo_wrapper();
}

struct SparseMvWrapper {
    const int* ia;
    const int* ja;
    const float* a;
    const float* x;
    float* res;
    __device__ void BlockReduce64(float* s_tmp_res, const float* s_data,
                                  float* res) {
        int warp_id = threadIdx.x >> 5;
        if (threadIdx.x < 2) {
            s_tmp_res[threadIdx.x] = 0.f;
        }

        float value = s_data[threadIdx.x];
        for (int i = 16; i >= 1; i >>= 1) {
            value += __shfl_xor_sync(__activemask(), value, i, 32);
        }
        // lane_id == 31, write result back.
        if ((threadIdx.x & 0x1f) == 31) {
            s_tmp_res[warp_id] = value;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            *res = s_tmp_res[0] + s_tmp_res[1];
        }
    }

    __device__ void operator()() {
        __shared__ float s_res_tmp[64];
        __shared__ int s_start_idx;
        __shared__ int s_num_row_nnz;
        __shared__ float s_reduce_tmp[2];

        if (threadIdx.x == 0) {
            s_start_idx   = ia[blockIdx.x];
            s_num_row_nnz = ia[blockIdx.x + 1] - s_start_idx;
        }
        __syncthreads();

        s_res_tmp[threadIdx.x] = 0.f;

        for (int iter = threadIdx.x; iter < s_num_row_nnz; iter += blockDim.x) {
            s_res_tmp[threadIdx.x] +=
                a[s_start_idx + iter] * x[ja[s_start_idx + iter]];
        }
        __syncthreads();

        BlockReduce64(s_reduce_tmp, s_res_tmp, &(res[blockIdx.x]));
    }
};
__global__ void SparseMvKernel(SparseMvWrapper spmv_wrapper) {
    spmv_wrapper();
}

PcgLinearSolverCPUGPU::PcgLinearSolverCPUGPU(int row)
    : cusparseHandle_(nullptr), descr_(nullptr), row_(row) {
    cusparseCreate(&cusparseHandle_);
    cusparseCreateMatDescr(&descr_);
    cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);

    d_buf_.resize(row_ * 2);
    d_p_     = RAW_PTR(d_buf_);
    d_omega_ = RAW_PTR(d_buf_) + row_;

    buf_.resize(row_ * 6);
    omega_   = buf_.data();
    p_       = buf_.data() + row_;
    r_       = buf_.data() + row_ * 2;
    z_       = buf_.data() + row_ * 3;
    delta_x_ = buf_.data() + row_ * 4;
    precond_ = buf_.data() + row_ * 5;

    spmv_wrapper_ = new SparseMvWrapper;
}

PcgLinearSolverCPUGPU::~PcgLinearSolverCPUGPU() {
    cusparseDestroy(cusparseHandle_);
    cusparseDestroyMatDescr(descr_);

    delete spmv_wrapper_;
}

void PcgLinearSolverCPUGPU::Solve(float* d_x, int* d_ia, int* d_ja, float* d_a,
                                  int num_nnz, float* d_b, float* precond,
                                  int max_iter) {
    spmv_wrapper_->ia  = d_ia;
    spmv_wrapper_->ja  = d_ja;
    spmv_wrapper_->a   = d_a;
    spmv_wrapper_->x   = d_p_;
    spmv_wrapper_->res = d_omega_;

    float *p1, *p2, *p3;
    checkCudaErrors(cudaMemcpy(r_, RAW_PTR(d_b), row_ * sizeof(float),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(precond_, RAW_PTR(precond), row_ * sizeof(float),
                               cudaMemcpyDeviceToHost));
    p1 = p_, p2 = precond_, p3 = r_;
    zr_dot_ = 0.0f;
    for (int i = 0; i < row_; ++i) {
        *p1 = (*(p2++)) * (*p3);
        zr_dot_ += (*(p1++)) * (*(p3++));
    }
    memset(delta_x_, 0, row_ * sizeof(float));
    double r_val;
    int iterCnt = 0;
    for (int k = 0; k < max_iter; k++) {
        zr_dot_old_ = zr_dot_;
        checkCudaErrors(
            cudaMemcpy(d_p_, p_, row_ * sizeof(float), cudaMemcpyHostToDevice));

        SparseMvKernel<<<row_, 64>>>(*spmv_wrapper_);
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(omega_, d_omega_, row_ * sizeof(float),
                                   cudaMemcpyDeviceToHost));

        p1 = p_, p2 = omega_;
        alpha_ = 0.0f;
        for (int i = 0; i < row_; ++i) {
            alpha_ += (*(p1++)) * (*(p2++));
        }
        alpha_ = zr_dot_ / alpha_;

        p1 = delta_x_, p2 = p_;
        for (int i = 0; i < row_; ++i) {
            *(p1++) += alpha_ * (*(p2++));
        }

        p1 = r_, p2 = omega_;
        r_val = 0.0;
        for (int i = 0; i < row_; ++i) {
            r_val += (*p1) * (*p1);
            *(p1++) -= alpha_ * (*(p2++));
        }

        p1 = z_, p2 = precond_, p3 = r_;
        for (int i = 0; i < row_; ++i) {
            *(p1) = (*(p2++)) * (*(p3++));
            ++p1;
        }

        p1 = r_, p2 = z_;
        zr_dot_ = 0.0f;
        for (int i = 0; i < row_; ++i) {
            zr_dot_ += (*(p1++)) * (*(p2++));
        }

        beta_ = zr_dot_ / zr_dot_old_;
        p1 = p_, p2 = z_;
        for (int i = 0; i < row_; ++i) {
            *(p1) = (*(p2++)) + beta_ * (*p1);
            ++p1;
        }
        ++iterCnt;
    }
    checkCudaErrors(cudaMemcpy(RAW_PTR(d_x), delta_x_, row_ * sizeof(float),
                               cudaMemcpyHostToDevice));
}

PcgLinearSolverGPU::PcgLinearSolverGPU(int row)
    : cusparseHandle_(nullptr), descr_(nullptr), row_(row) {
    cusparseCreate(&cusparseHandle_);
    cusparseCreateMatDescr(&descr_);
    cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);

    d_buf_.resize(row_ * 5);
    d_omega_   = RAW_PTR(d_buf_);
    d_p_       = RAW_PTR(d_buf_) + row_;
    d_r_       = RAW_PTR(d_buf_) + row_ * 2;
    d_z_       = RAW_PTR(d_buf_) + row_ * 3;
    d_delta_x_ = RAW_PTR(d_buf_) + row_ * 4;

    d_zr_dot_.resize(1);
    d_zr_dot_old_.resize(1);
    d_alpha_.resize(1);
    d_beta_.resize(1);

    dm_wrapper_       = new DotMuliplyWrapper;
    spmv_wrapper_     = new SparseMvWrapper;
    ip_wrapper_       = new InnerProductWrapper;
    ip_wrapper_alpha_ = new InnerProductWrapper;
    ao_wrapper_       = new AxpyOperatorWrapper;
    uzar_wrapper_     = new UpdateZAndRWrapper;
    ip_wrapper_beta_  = new InnerProductWrapper;
    xo_wrapper_       = new XpayOperatorWrapper;
}

PcgLinearSolverGPU::~PcgLinearSolverGPU() {
    cusparseDestroy(cusparseHandle_);
    cusparseDestroyMatDescr(descr_);

    delete dm_wrapper_;
    delete spmv_wrapper_;
    delete ip_wrapper_;
    delete ip_wrapper_alpha_;
    delete ao_wrapper_;
    delete uzar_wrapper_;
    delete ip_wrapper_beta_;
    delete xo_wrapper_;
}

void PcgLinearSolverGPU::Solve(float* d_x, int* d_ia, int* d_ja, float* d_a,
                               int num_nnz, float* d_b, float* precond,
                               int max_iter) {
    checkCudaErrors(cudaMemcpy(RAW_PTR(d_r_), RAW_PTR(d_b),
                               sizeof(float) * row_, cudaMemcpyDeviceToDevice));
    int block_size           = 256;
    int grid_size            = (row_ + block_size - 1) / block_size;
    dm_wrapper_->num_data    = row_;
    dm_wrapper_->input_data1 = RAW_PTR(precond);
    dm_wrapper_->input_data2 = RAW_PTR(d_r_);
    dm_wrapper_->output      = RAW_PTR(d_p_);
    DotMuliplyKernel<<<grid_size, block_size>>>(*dm_wrapper_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    spmv_wrapper_->ia  = d_ia;
    spmv_wrapper_->ja  = d_ja;
    spmv_wrapper_->a   = d_a;
    spmv_wrapper_->x   = RAW_PTR(d_p_);
    spmv_wrapper_->res = RAW_PTR(d_omega_);

    ip_wrapper_->num_data    = row_;
    ip_wrapper_->input_data1 = RAW_PTR(d_r_);
    ip_wrapper_->input_data2 = RAW_PTR(d_p_);
    ip_wrapper_->output      = RAW_PTR(d_zr_dot_);
    InnerProductKernel<<<1, 1024>>>(*ip_wrapper_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    ip_wrapper_alpha_->num_data    = row_;
    ip_wrapper_alpha_->input_data1 = RAW_PTR(d_p_);
    ip_wrapper_alpha_->input_data2 = RAW_PTR(d_omega_);
    ip_wrapper_alpha_->output      = RAW_PTR(d_alpha_);

    ao_wrapper_->num_data = row_;
    ao_wrapper_->vector_x = RAW_PTR(d_p_);
    ao_wrapper_->vector_y = RAW_PTR(d_x);
    ao_wrapper_->scale_a  = RAW_PTR(d_alpha_);

    uzar_wrapper_->num_data       = row_;
    uzar_wrapper_->preconditioner = RAW_PTR(precond);
    uzar_wrapper_->omega          = RAW_PTR(d_omega_);
    uzar_wrapper_->d_r            = RAW_PTR(d_r_);
    uzar_wrapper_->d_z            = RAW_PTR(d_z_);
    uzar_wrapper_->scale_a        = RAW_PTR(d_alpha_);

    ip_wrapper_beta_->num_data    = row_;
    ip_wrapper_beta_->input_data1 = RAW_PTR(d_z_);
    ip_wrapper_beta_->input_data2 = RAW_PTR(d_r_);
    ip_wrapper_beta_->output      = RAW_PTR(d_zr_dot_);

    xo_wrapper_->num_data = row_;
    xo_wrapper_->vector_x = RAW_PTR(d_z_);
    xo_wrapper_->vector_y = RAW_PTR(d_p_);
    xo_wrapper_->scale_a  = RAW_PTR(d_beta_);

    float zr_dot;
    for (int k = 0; k < max_iter; ++k) {
        checkCudaErrors(cudaMemcpy(RAW_PTR(d_zr_dot_old_), RAW_PTR(d_zr_dot_),
                                   sizeof(float), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(&zr_dot, RAW_PTR(d_zr_dot_), sizeof(float),
                                   cudaMemcpyDeviceToHost));
        if (zr_dot < 1.0e-5f)
            break;

        SparseMvKernel<<<row_, 64>>>(*spmv_wrapper_);

        // update alpha_
        InnerProductKernel<<<1, 1024>>>(*ip_wrapper_alpha_);
        UpdateAlphaKernel<<<1, 1>>>(RAW_PTR(d_alpha_), RAW_PTR(d_zr_dot_));

        // update x
        AxpyOperatorKernel<<<grid_size, block_size>>>(*ao_wrapper_);

        // update r and z
        UpdateZAndRKernel<<<grid_size, block_size>>>(*uzar_wrapper_);

        // update beta_
        InnerProductKernel<<<1, 1024>>>(*ip_wrapper_beta_);
        UpdataBetaKernel<<<1, 1>>>(RAW_PTR(d_zr_dot_), RAW_PTR(d_zr_dot_old_),
                                   RAW_PTR(d_beta_));

        // update p
        XapyOperatorKernel<<<grid_size, block_size>>>(*xo_wrapper_);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}
