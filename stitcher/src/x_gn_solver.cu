#include "constraint_terms.cuh"
#include "cuda_utils.h"
#include "helper_cuda.h"
#include "innoreal_timer.hpp"
#include "math_utils.h"
#include "nz_block_statistics.cuh"
#include "reduce.cuh"
#include "x_gn_solver.cuh"
#include "x_pcg_solver.cuh"
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#define USE_CHOLSKY 0

GNSolver::GNSolver(int img_rows, int img_cols, int node_img_rows,
                   int node_img_cols, int* d_pixel_rela_idx_vec,
                   float* d_pixel_rela_weight_vec, int* d_node_rela_idx_vec,
                   float* d_node_rela_weight_vec, float2* d_original_node_vec,
                   float2* d_node_vec, cv::cuda::GpuMat& d_src_img,
                   cv::cuda::GpuMat& d_target_img, cv::cuda::GpuMat& d_dy_img,
                   cv::cuda::GpuMat& d_dx_img, float triangle_u,
                   float triangle_v, const SolverPara& para)
    : img_rows_(img_rows), img_cols_(img_cols), node_img_rows_(node_img_rows),
      node_img_cols_(node_img_cols),
      d_pixel_rela_idx_vec_(d_pixel_rela_idx_vec),
      d_pixel_rela_weight_vec_(d_pixel_rela_weight_vec),
      d_node_rela_idx_vec_(d_node_rela_idx_vec),
      d_node_rela_weight_vec_(d_node_rela_weight_vec),
      d_original_node_vec_(d_original_node_vec), d_node_vec_(d_node_vec),
      d_src_img_(d_src_img), d_target_img_(d_target_img), d_dy_img_(d_dy_img),
      d_dx_img_(d_dx_img), triangle_u_(triangle_u), triangle_v_(triangle_v),
      para_(para) {
    pixel_num_    = img_rows * img_cols;
    triangle_num_ = (node_img_rows - 1) * (node_img_cols - 1) * 2;
    node_num_     = node_img_rows * node_img_cols;

    if (para.has_data_term_) {
        DataTermConstraint* p_term_cons = new DataTermConstraint;
        p_term_cons->Init(this, para.data_term_weight_);
        cons_.push_back(p_term_cons);
    }
    if (para.has_smooth_term_) {
        SmoothTermConstraint* p_term_cons = new SmoothTermConstraint;
        p_term_cons->Init(this, para.smooth_term_weight_);
        cons_.push_back(p_term_cons);
    }
    if (para.has_zero_term_) {
        ZeroTermConstraint* p_term_cons = new ZeroTermConstraint;
        p_term_cons->Init(this, para.zero_term_weight_);
        cons_.push_back(p_term_cons);
    }

    vars_num_ = node_num_ * 2;
    d_delta_.resize(vars_num_);
    d_JTb_.resize(vars_num_);
    d_preconditioner_.resize(vars_num_);
    d_JTJ_ = new SparseMatrixCSR(vars_num_, vars_num_);

    /** 计算Iij集合 */
    nz_blocks_static_ = new NzBlockStatisticsForJTJ;
    nz_blocks_static_->CalcNzBlocksAndFunctions(
        pixel_num_, triangle_num_, node_num_, d_pixel_rela_idx_vec,
        d_pixel_rela_weight_vec, d_node_rela_idx_vec, d_node_rela_weight_vec);

    /** 填充CSR格式，jtj_ia和jtj_ja */
    InitJTJ();
    pcg_linear_solver_ = new PcgLinearSolverGPU(vars_num_);
}

struct InitJaOfJTJWrapper {
    int* jtj_ja;

    int node_num;
    int nz_block_num;
    int* nz_block_idx;
    int* pre_nz_block_num;
    int* row_offset;

    __device__ void operator()() {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < nz_block_num) {
            // 填充下三角中的非零块
            int index            = nz_block_idx[idx];
            int seri_i           = index / node_num;
            int seri_j           = index - seri_i * node_num;
            int num_pre_row      = pre_nz_block_num[index];
            int num_pre_all      = row_offset[seri_i];
            int num_nnz_this_row = row_offset[seri_i + 1] - num_pre_all;
            for (int iter_row = 0; iter_row < 2; ++iter_row) {
                for (int iter_col = 0; iter_col < 2; ++iter_col) {
                    jtj_ja[num_pre_all * 4 + iter_row * num_nnz_this_row * 2 +
                           num_pre_row * 2 + iter_col] = seri_j * 2 + iter_col;
                }
            }
            // 填充上三角中的非零块
            int tmp          = seri_i;
            seri_i           = seri_j;
            seri_j           = tmp;
            index            = seri_i * node_num + seri_j;
            num_pre_row      = pre_nz_block_num[index];
            num_pre_all      = row_offset[seri_i];
            num_nnz_this_row = row_offset[seri_i + 1] - num_pre_all;
            for (int iter_row = 0; iter_row < 2; ++iter_row) {
                for (int iter_col = 0; iter_col < 2; ++iter_col) {
                    jtj_ja[num_pre_all * 4 + iter_row * num_nnz_this_row * 2 +
                           num_pre_row * 2 + iter_col] = seri_j * 2 + iter_col;
                }
            }
        }
    }
};
__global__ void InitJaOfJTJKernel(InitJaOfJTJWrapper ijoj_wrapper) {
    ijoj_wrapper();
}

void GNSolver::InitJTJ() {
    int all_nz_block_num =
        nz_blocks_static_->nz_block_num_ * 2 -
        node_num_; /** 所有的非零元素个数，nz_block_num_为下三角非零元素个数 */
    d_JTJ_->row_ = node_num_ * 2;
    d_JTJ_->col_ = node_num_ * 2;
    d_JTJ_->nnz_ = all_nz_block_num * 4;
    d_JTJ_->d_ja_.resize(d_JTJ_->nnz_);
    d_JTJ_->d_ia_.resize(node_num_ * 2 + 1);
    d_JTJ_->d_a_.clear();
    d_JTJ_->d_a_.resize(d_JTJ_->nnz_, 1.0f);

    InitJaOfJTJWrapper ijoj_wrapper;
    ijoj_wrapper.jtj_ja       = RAW_PTR(d_JTJ_->d_ja_);
    ijoj_wrapper.node_num     = node_num_;
    ijoj_wrapper.nz_block_num = nz_blocks_static_->nz_block_num_;
    ijoj_wrapper.nz_block_idx = RAW_PTR(nz_blocks_static_->d_nz_block_idx_vec_);
    ijoj_wrapper.pre_nz_block_num =
        RAW_PTR(nz_blocks_static_->d_pre_nz_block_num_vec_);
    ijoj_wrapper.row_offset = RAW_PTR(nz_blocks_static_->d_row_offset_vec_);
    int block               = 256,
        grid = (nz_blocks_static_->nz_block_num_ + block - 1) / block;
    InitJaOfJTJKernel<<<grid, block>>>(ijoj_wrapper);

    thrust::host_vector<int> ia(node_num_ * 2 + 1);
    thrust::host_vector<int> row_offset = nz_blocks_static_->d_row_offset_vec_;
    int offset                          = 0;
    int nnz_each_row;
    int counter = 0;
    for (int iter = 0; iter < node_num_; ++iter) {
        nnz_each_row = (row_offset[iter + 1] - row_offset[iter]) * 2;
        for (int iter_inner = 0; iter_inner < 2; ++iter_inner) {
            ia[counter] = offset;
            ++counter;
            offset += nnz_each_row;
        }
    }
    ia[counter]   = offset;
    d_JTJ_->d_ia_ = ia;
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

__global__ void ExtractPreconditioner(float* preCondTerms, int* ia, int* ja,
                                      float* a, int rowJTJ) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= rowJTJ) {
        return;
    }

    for (int i = ia[idx]; i < ia[idx + 1]; ++i) {
        if (idx == ja[i]) {
            preCondTerms[idx] = 1.0f / a[i];
            return;
        }
    }
}

__global__ void UpdateNodesKernel(float* node_vec, int node_vec_len,
                                  float* delta) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= node_vec_len) {
        return;
    }

    node_vec[idx] += delta[idx];
}

void GNSolver::UpdateNodes() {
    int block = 512, grid = (vars_num_ + block - 1) / block;
    UpdateNodesKernel<<<grid, block>>>((float*)RAW_PTR(d_node_vec_), vars_num_,
                                       RAW_PTR(d_delta_));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * @brief 计算-warped pixels的内核函数
 *
 * 该函数用于根据给定的节点、像素关系及其权重，计算出一个warped pixel向量。
 *
 * @param warped_pixel_vec 输出的warped pixel向量
 * @param pixel_num 像素数量
 * @param img_cols 图像列数
 * @param node_vec 节点向量
 * @param pixel_rela_idx_vec 像素关系索引向量
 * @param pixel_rela_weight_vec 像素关系权重向量
 * @param src_rows 源图像行数
 * @param src_cols 源图像列数
 * @param target_rows 目标图像行数
 * @param target_cols 目标图像列数
 */
__global__ void CalcWarpedPixelsKernel(float2* warped_pixel_vec, int pixel_num,
                                       int img_cols, float2* node_vec,
                                       int* pixel_rela_idx_vec,
                                       float* pixel_rela_weight_vec,
                                       int src_rows, int src_cols,
                                       int target_rows, int target_cols) {
    // 计算当前处理的像素索引
    int pixel_idx = threadIdx.x + blockDim.x * blockIdx.x;
    // 如果索引超出像素数量范围，则返回
    if (pixel_idx >= pixel_num) {
        return;
    }

    // 根据索引计算像素的行和列
    int row = pixel_idx / img_cols;
    int col = pixel_idx % img_cols;
    // 计算四个方向的像素关系索引和权重的偏移量
    int tl = 4 * pixel_idx;
    int tr = 4 * pixel_idx + 1;
    int bl = 4 * pixel_idx + 2;
    int br = 4 * pixel_idx + 3;

    // 根据像素关系和权重计算warped pixel
    float2 warped_pixel =
        pixel_rela_weight_vec[tl] * node_vec[pixel_rela_idx_vec[tl]] +
        pixel_rela_weight_vec[tr] * node_vec[pixel_rela_idx_vec[tr]] +
        pixel_rela_weight_vec[bl] * node_vec[pixel_rela_idx_vec[bl]] +
        pixel_rela_weight_vec[br] * node_vec[pixel_rela_idx_vec[br]];

    // 如果warped pixel在有效范围内，则保存，否则标记为无效
    if (warped_pixel.y >= 0 && warped_pixel.x >= 0 &&
        warped_pixel.y < src_rows && warped_pixel.x < src_cols) {
        warped_pixel_vec[pixel_idx] = warped_pixel;
    } else {
        warped_pixel_vec[pixel_idx] = make_float2(-1.0f, -1.0f);
    }
}

/**
 * 计算变形像素
 * 本函数用于根据当前的节点、像素关系及权重等信息，计算出变形后的像素值。
 * 此函数会调用 CUDA 核函数 `CalcWarpedPixelsKernel` 在 GPU 上进行计算。
 *
 * 参数:
 * - 无
 *
 * 返回值:
 * - 无
 */

void GNSolver::CalcWarpedPixels() {
    // 检查并调整变形像素向量的大小，确保能够容纳所有像素
    if (pixel_num_ > d_warped_pixel_vec_.size()) {
        d_warped_pixel_vec_.resize(pixel_num_);
    }

    // 设置 CUDA kernel 调用的网格和块大小
    int block = 128, grid = (pixel_num_ + block - 1) / block;

    // 调用 CUDA kernel 函数进行变形像素的计算
    CalcWarpedPixelsKernel<<<grid, block>>>(
        RAW_PTR(d_warped_pixel_vec_), pixel_num_, img_cols_,
        RAW_PTR(d_node_vec_), RAW_PTR(d_pixel_rela_idx_vec_),
        RAW_PTR(d_pixel_rela_weight_vec_), d_src_img_.rows, d_src_img_.cols,
        d_target_img_.rows, d_target_img_.cols);

    // 确保所有 CUDA 操作完成，检查有无错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 尝试进行下一步的求解过程。
 * 该函数不接受参数，返回一个布尔值，表示是否成功进行了下一步的求解。
 *
 * 主要步骤包括：
 * 1. 计算变形像素。
 * 2. 清零delta、JTb和JTJ矩阵的存储空间。
 * 3. 累加JTJ和JTb。
 * 4. 使用PCG线性求解器解决方程。
 * 5. 更新节点位置。
 *
 * @return 总是返回true，表示求解过程可以继续。
 */
bool GNSolver::Next() {
    CalcWarpedPixels();   // 计算变形像素

    // 使用cudaMemset清零delta、JTb和JTJ的存储空间
    checkCudaErrors(
        cudaMemset(RAW_PTR(d_delta_), 0, d_delta_.size() * sizeof(float)));
    checkCudaErrors(
        cudaMemset(RAW_PTR(d_JTb_), 0, d_JTb_.size() * sizeof(float)));
    checkCudaErrors(cudaMemset(RAW_PTR(d_JTJ_->d_a_), 0,
                               d_JTJ_->d_a_.size() * sizeof(float)));

    // 遍历约束，累加JTJ和JTb
    std::vector<Constraint*>::iterator it;
    for (it = cons_.begin(); it != cons_.end(); ++it) {
        Constraint* pCon = *it;
        pCon->GetJTJAndJTb(RAW_PTR(d_JTJ_->d_a_), RAW_PTR(d_JTJ_->d_ia_),
                           RAW_PTR(d_JTb_), (float*)RAW_PTR(d_node_vec_));
    }

    // 设置CUDAKernel的块和网格大小，并执行ExtractPreconditioner kernel
    int block = 512, grid = (vars_num_ + block - 1) / block;
    ExtractPreconditioner<<<grid, block>>>(
        RAW_PTR(d_preconditioner_), RAW_PTR(d_JTJ_->d_ia_),
        RAW_PTR(d_JTJ_->d_ja_), RAW_PTR(d_JTJ_->d_a_), vars_num_);

    // 使用PCG求解器解决线性方程组
    pcg_linear_solver_->Solve(RAW_PTR(d_delta_), RAW_PTR(d_JTJ_->d_ia_),
                              RAW_PTR(d_JTJ_->d_ja_), RAW_PTR(d_JTJ_->d_a_),
                              d_JTJ_->nnz_, RAW_PTR(d_JTb_),
                              RAW_PTR(d_preconditioner_), 8);

    UpdateNodes();   // 更新节点位置

    return true;
}
