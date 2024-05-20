#include "calc/constraint_terms.cuh"
#include "calc/nz_block_statistics.cuh"
#include "calc/reduce.cuh"
#include "calc/x_gn_solver.cuh"
#include "calc/x_pcg_solver.cuh"
#include "util/cuda_utils.h"
#include "util/helper_cuda.h"
#include "util/innoreal_timer.hpp"
#include "util/math_utils.h"
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#define USE_CHOLSKY 0

/**
 * GNSolver的构造函数
 * 用于初始化GNSolver对象，包括设置图像尺寸、节点尺寸、各种向量和矩阵的设备内存指针，以及初始化约束条件和求解器参数。
 *
 * @param img_rows 输入图像的行数
 * @param img_cols 输入图像的列数
 * @param node_img_rows 节点图像的行数
 * @param node_img_cols 节点图像的列数
 * @param d_pixel_rela_idx_vec 设备上像素关系索引向量的指针
 * @param d_pixel_rela_weight_vec 设备上像素关系权重向量的指针
 * @param d_node_rela_idx_vec 设备上节点关系索引向量的指针
 * @param d_node_rela_weight_vec 设备上节点关系权重向量的指针
 * @param d_original_node_vec 设备上原始节点向量的指针
 * @param d_node_vec 设备上节点向量的指针
 * @param d_src_img 输入图像的设备内存矩阵
 * @param d_target_img 目标图像的设备内存矩阵
 * @param d_dy_img 输入图像的梯度Y方向的设备内存矩阵
 * @param d_dx_img 输入图像的梯度X方向的设备内存矩阵
 * @param triangle_u 三角形的u参数
 * @param triangle_v 三角形的v参数
 * @param para 求解器参数
 */
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
    // 初始化基本变量
    pixel_num_    = img_rows * img_cols;
    triangle_num_ = (node_img_rows - 1) * (node_img_cols - 1) * 2;
    node_num_     = node_img_rows * node_img_cols;

    // 根据求解器参数初始化不同的约束条件
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

    // 初始化变量数量及相关矩阵
    vars_num_ = node_num_ * 2;
    d_delta_.resize(vars_num_);
    d_JTb_.resize(vars_num_);
    d_preconditioner_.resize(vars_num_);
    d_JTJ_ = new SparseMatrixCSR(vars_num_, vars_num_);

    // 计算非零块统计信息，用于稀疏矩阵乘法
    nz_blocks_static_ = new NzBlockStatisticsForJTJ;
    nz_blocks_static_->CalcNzBlocksAndFunctions(
        pixel_num_, triangle_num_, node_num_, d_pixel_rela_idx_vec,
        d_pixel_rela_weight_vec, d_node_rela_idx_vec, d_node_rela_weight_vec);

    // 初始化JTJ矩阵的CSR表示
    InitJTJ();
    // 初始化PCG线性求解器
    pcg_linear_solver_ = new PcgLinearSolverGPU(vars_num_);
}

/**
 * 结构体 InitJaOfJTJWrapper 用于初始化 jtj_ja 数组，
 * jtj_ja 数组表示一个下三角和对角线部分的索引，该操作针对 CUDA 设备进行。
 */
struct InitJaOfJTJWrapper {
    int* jtj_ja;   // 存储下三角和对角线部分的索引数组

    int node_num;            // 节点数量
    int nz_block_num;        // 非零块的数量
    int* nz_block_idx;       // 非零块的索引数组
    int* pre_nz_block_num;   // 每个非零块前的非零元素数量
    int* row_offset;   // 每行的偏移量，用于计算行内非零元素的起始位置

    /**
     * 在 CUDA 设备上运行的运算符()函数，用于填充 jtj_ja 数组。
     * 填充规则为下三角和对角线部分，同时镜像填充上三角。
     */
    __device__ void operator()() {
        int idx =
            threadIdx.x + blockIdx.x * blockDim.x;   // 计算当前线程处理的索引
        if (idx < nz_block_num) {   // 检查索引是否在非零块的数量范围内
            // 填充下三角中的非零块
            int index  = nz_block_idx[idx];
            int seri_i = index / node_num;            // 计算行索引
            int seri_j = index - seri_i * node_num;   // 计算列索引
            int num_pre_row =
                pre_nz_block_num[index];   // 当前行前的非零元素数量
            int num_pre_all = row_offset[seri_i];   // 当前行起始的非零元素数量
            int num_nnz_this_row = row_offset[seri_i + 1] -
                                   num_pre_all;   // 当前行内的非零元素数量
            for (int iter_row = 0; iter_row < 2;
                 ++iter_row) {   // 遍历当前块的行
                for (int iter_col = 0; iter_col < 2;
                     ++iter_col) {   // 遍历当前块的列
                    jtj_ja[num_pre_all * 4 + iter_row * num_nnz_this_row * 2 +
                           num_pre_row * 2 + iter_col] = seri_j * 2 + iter_col;
                }
            }
            // 填充上三角中的非零块，通过交换行和列来实现
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

/**
 * CUDA 核函数 InitJaOfJTJKernel 用于在 GPU 上初始化 jtj_ja 数组。
 * @param ijoj_wrapper 包含所有必要数据和操作的 InitJaOfJTJWrapper 结构体实例。
 */
__global__ void InitJaOfJTJKernel(InitJaOfJTJWrapper ijoj_wrapper) {
    ijoj_wrapper();   // 调用 InitJaOfJTJWrapper 的运算符()函数完成初始化
}

/**
 * 初始化JTJ矩阵的结构信息和非零元素数组。
 * JTJ矩阵用于数值求解等过程中，此处主要完成矩阵存储结构的初始化。
 *
 * 无参数
 * 无返回值
 */
void GNSolver::InitJTJ() {
    // 计算JTJ矩阵的所有非零元素数量
    int all_nz_block_num =
        nz_blocks_static_->nz_block_num_ * 2 -
        node_num_; /** 所有的非零元素个数，基于下三角非零元素个数计算 */
    d_JTJ_->row_ = node_num_ * 2;          // 矩阵行数
    d_JTJ_->col_ = node_num_ * 2;          // 矩阵列数
    d_JTJ_->nnz_ = all_nz_block_num * 4;   // 矩阵非零元素个数
    // 分配存储空间
    d_JTJ_->d_ja_.resize(d_JTJ_->nnz_);
    d_JTJ_->d_ia_.resize(node_num_ * 2 + 1);
    d_JTJ_->d_a_.clear();
    d_JTJ_->d_a_.resize(d_JTJ_->nnz_, 1.0f);   // 初始化非零元素值为1

    // 准备kernel函数调用所需的参数
    InitJaOfJTJWrapper ijoj_wrapper;
    ijoj_wrapper.jtj_ja       = RAW_PTR(d_JTJ_->d_ja_);
    ijoj_wrapper.node_num     = node_num_;
    ijoj_wrapper.nz_block_num = nz_blocks_static_->nz_block_num_;
    ijoj_wrapper.nz_block_idx = RAW_PTR(nz_blocks_static_->d_nz_block_idx_vec_);
    ijoj_wrapper.pre_nz_block_num =
        RAW_PTR(nz_blocks_static_->d_pre_nz_block_num_vec_);
    ijoj_wrapper.row_offset = RAW_PTR(nz_blocks_static_->d_row_offset_vec_);
    // 调用CUDA kernel函数计算JTJ矩阵的ja数组
    int block = 256,
        grid  = (nz_blocks_static_->nz_block_num_ + block - 1) / block;
    InitJaOfJTJKernel<<<grid, block>>>(ijoj_wrapper);

    // 在主机端创建临时向量用于构建ia数组
    thrust::host_vector<int> ia(node_num_ * 2 + 1);
    thrust::host_vector<int> row_offset = nz_blocks_static_->d_row_offset_vec_;
    int offset                          = 0;
    int nnz_each_row;
    int counter = 0;
    // 计算并填充ia数组，用于索引JTJ矩阵的每一行的起始位置
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
    // 确保所有CUDA操作完成，检查是否有错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 在GPU上执行矩阵预条件项的提取。
 *
 * @param preCondTerms 用于存储预条件项的浮点数数组。
 * @param ia 存储矩阵稀疏表示的行索引数组。
 * @param ja 存储矩阵稀疏表示的列索引数组。
 * @param a 存储矩阵稀疏表示的值数组。
 * @param rowJTJ J'TJ矩阵的行数。
 *
 * 此函数旨在针对大规模稀疏矩阵问题进行预处理项的并行计算，
 * 适用于预处理技术中的求逆矩阵近似等操作。
 */
__global__ void ExtractPreconditioner(float* preCondTerms, int* ia, int* ja,
                                      float* a, int rowJTJ) {
    // 根据线程索引和块索引计算当前线程处理的数据索引
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 如果当前索引超出矩阵范围，则提前退出
    if (idx >= rowJTJ) {
        return;
    }

    // 遍历当前行的所有非零元素，寻找对应的预条件项
    for (int i = ia[idx]; i < ia[idx + 1]; ++i) {
        // 如果找到当前线程负责的列，则计算其预条件项并返回
        if (idx == ja[i]) {
            preCondTerms[idx] = 1.0f / a[i];
            return;
        }
    }
}

/**
 * 在GPU上更新节点向量的值。
 *
 * @param node_vec 指向需要更新的节点向量的指针。
 * @param node_vec_len 节点向量的长度。
 * @param delta 指向包含更新增量的向量的指针。
 *
 * 此函数通过线程块和线程索引的组合来并行更新节点向量中的元素，每个元素增加对应的增量值。
 */
__global__ void UpdateNodesKernel(float* node_vec, int node_vec_len,
                                  float* delta) {
    // 计算当前线程处理的节点索引
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // 如果索引超出节点向量长度，则提前返回
    if (idx >= node_vec_len) {
        return;
    }

    // 更新节点向量中的值
    node_vec[idx] += delta[idx];
}

/**
 * 更新节点信息
 * 此函数用于通过调用CUDA内核函数UpdateNodesKernel来更新节点的状态。
 * 它首先根据变量数量计算出需要的CUDA网格和块的数量，然后启动UpdateNodesKernel，
 * 并检查CUDA执行过程中是否有错误发生。
 */
void GNSolver::UpdateNodes() {
    // 计算CUDA执行网格和块的大小
    int block = 512, grid = (vars_num_ + block - 1) / block;
    // 调用CUDA内核函数UpdateNodesKernel以更新节点
    UpdateNodesKernel<<<grid, block>>>((float*)RAW_PTR(d_node_vec_), vars_num_,
                                       RAW_PTR(d_delta_));
    // 等待所有CUDA任务完成，并检查是否有错误
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
