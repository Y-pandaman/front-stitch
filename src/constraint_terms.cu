#include "constraint_terms.cuh"
#include "cuda_utils.h"
#include "helper_cuda.h"
#include "reduce.cuh"
#include "x_gn_solver.cuh"
#include <device_launch_parameters.h>

/**
 * 初始化数据项约束。
 * 该函数为数据项约束初始化过程，负责设置相关的求解器指针、行列数、以及权重。
 *
 * @param gn_solver 指向GNSolver对象的指针，用于获取和设置求解器的相关参数。
 * @param weight 权重参数，用于数据项约束的加权。
 * @return 总是返回true，表示初始化成功。
 */
bool DataTermConstraint::Init(GNSolver* gn_solver, float weight) {
    assert(gn_solver);        // 确保gn_solver不为nullptr。
    gn_solver_ = gn_solver;   // 存储GNSolver对象的指针。
    row_ = gn_solver_->pixel_num_;   // 设置行数，基于gn_solver的像素数量。
    col_ = gn_solver_->vars_num_;   // 设置列数，基于gn_solver的变量数量。
    d_b_.resize(row_);              // 根据行数调整d_b_向量的大小。
    SetWeight(weight);              // 设置权重。

    return true;
}

bool DataTermConstraint::Init() {
    row_ = gn_solver_->pixel_num_;
    col_ = gn_solver_->vars_num_;
    d_b_.resize(row_);

    return true;
}

/**
 * 计算并获取数据项的JTJ矩阵和JTb向量。
 * @param:
 *   - d_JTJ_a: 指向浮点数数组的指针，用于存储JTJ矩阵的值。
 *   - d_JTJ_ia: 指向整数数组的指针，用于存储JTJ矩阵的索引。
 *   - d_JTb: 指向浮点数数组的指针，用于存储JTb向量的值。
 *   - d_x: 指向浮点数数组的指针，输入参数，表示当前的解估计。
 */
void DataTermConstraint::GetJTJAndJTb(float* d_JTJ_a, int* d_JTJ_ia,
                                      float* d_JTb, float* d_x) {
    // 应用当前解估计，更新b值
    b(d_x);
    // 计算并填充JTJ矩阵的值和索引
    DirectiveJTJ(d_JTJ_a, d_JTJ_ia);
    // 计算并填充JTb向量的值
    DirectiveJTb(d_JTb);
}

/**
 * 在GPU上计算数据项残差的内核函数。
 *
 * @param b 存储残差的结果数组，其大小为像素数量。
 * @param pixel_num 图像中像素的数量。
 * @param rows 输入图像的行数。
 * @param cols 输入图像的列数。
 * @param blur_src_img
 * 模糊的源图像的设备内存指针，使用cv::cuda::PtrStepSz封装以提供行步长和图像尺寸信息。
 * @param blur_target_img
 * 模糊的目标图像的设备内存指针，同样使用cv::cuda::PtrStepSz封装。
 * @param warped_pixel_vec
 * 已经映射（变形）的像素点的数组，其大小为像素数量，每个元素为float2。
 * @param weight 残差的权重因子。
 *
 * 该函数遍历所有像素，计算源图像和目标图像在给定已变形像素位置的强度差异，并根据这一差异更新结果数组b。
 */
__global__ void
CalcDataTermResidualKernel(float* b, int pixel_num, int rows, int cols,
                           cv::cuda::PtrStepSz<float> blur_src_img,
                           cv::cuda::PtrStepSz<float> blur_target_img,
                           float2* warped_pixel_vec, float weight) {
    // 计算当前线程处理的像素索引
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // 超出图像范围则返回
    if (idx >= pixel_num)
        return;

    // 根据索引计算像素的行列位置
    int row = idx / cols;
    int col = idx % cols;

    // 获取已变形像素的具体位置
    float2 warped_pixel = warped_pixel_vec[idx];
    // 使用双线性插值获取源图像在已变形像素位置的强度值
    float src_intensity = GetPixelValueBilinearDevice(
        blur_src_img, warped_pixel.x, warped_pixel.y);
    // 直接从目标图像获取对应位置的强度值
    float target_intensity = blur_target_img.ptr(row)[col];

    // 如果源或目标图像的强度值异常，则将残差设为0
    if (target_intensity > 2.0f || src_intensity > 2.0f) {
        b[idx] = 0;
        return;
    }
    // 计算并更新残差
    b[idx] = weight * (src_intensity - target_intensity);
}

/**
 * 为数据项约束项设置b向量。
 * 该函数首先清零b向量，然后通过调用CUDA内核计算数据项残差，更新b向量。
 *
 * @param d_x 指向设备上b向量的指针。
 */
void DataTermConstraint::b(float* d_x) {
    // 使用cudaMemset清零设备上b向量
    checkCudaErrors(
        cudaMemset(RAW_PTR(d_b_), 0, gn_solver_->pixel_num_ * sizeof(float)));

    // 计算CUDA kernel执行时的网格和块大小
    int block = 128, grid = (gn_solver_->pixel_num_ + block - 1) / block;

    // 调用CUDA kernel计算数据项残差，并更新b向量
    CalcDataTermResidualKernel<<<grid, block>>>(
        RAW_PTR(d_b_), gn_solver_->pixel_num_, gn_solver_->img_rows_,
        gn_solver_->img_cols_, gn_solver_->d_src_img_,
        gn_solver_->d_target_img_, RAW_PTR(gn_solver_->d_warped_pixel_vec_),
        weight_);

    // 确保kernel执行完成，检查CUDA错误
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 计算数据项的JTJ矩阵
 *
 * 本函数用于计算光流估计过程中的数据项的JTJ矩阵，该矩阵用于后续的光流优化过程。
 * JTJ矩阵反映了图像像素变化与光流估计之间的关系。
 *
 * @param d_JTJ_a 指向JTJ矩阵浮点数数组的指针，该数组在GPU内存中。
 * @param d_JTJ_ia
 * 指向JTJ矩阵索引数组的指针，该数组在GPU内存中，用于存储矩阵非零元素的索引。
 *
 * 注意：函数不返回任何值，但会通过CUDA错误检查确保执行过程无误。
 */
void DataTermConstraint::DirectiveJTJ(float* d_JTJ_a, int* d_JTJ_ia) {
    // 计算数据项的JTJ矩阵
    CalcDataTermJTJ(d_JTJ_a, d_JTJ_ia, gn_solver_->d_dx_img_,
                    gn_solver_->d_dy_img_, *gn_solver_->nz_blocks_static_,
                    RAW_PTR(gn_solver_->d_warped_pixel_vec_),
                    gn_solver_->d_pixel_rela_idx_vec_,
                    gn_solver_->d_pixel_rela_weight_vec_,
                    gn_solver_->d_node_vec_, gn_solver_->img_rows_,
                    gn_solver_->img_cols_, gn_solver_->node_num_, weight_);

    // 检查CUDA执行过程是否出错，确保设备同步
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 计算数据项约束的Jacobian乘以b的项
 *
 * 本函数用于计算光流估计中的数据项约束的Jacobian乘以b的项，这是光流优化过程中的一个步骤。
 * 具体地，它通过调用`CalcDataTermJTb`来完成计算，并检查CUDA执行过程中的错误。
 *
 * @param d_JTb 指向CUDA设备内存中存放Jacobian乘以b的结果的指针。
 */
void DataTermConstraint::DirectiveJTb(float* d_JTb) {
    // 调用CalcDataTermJTb函数进行计算，传入所有必要的参数
    CalcDataTermJTb(d_JTb, RAW_PTR(d_b_), gn_solver_->d_dx_img_,
                    gn_solver_->d_dy_img_, *gn_solver_->nz_blocks_static_,
                    RAW_PTR(gn_solver_->d_warped_pixel_vec_),
                    gn_solver_->d_pixel_rela_idx_vec_,
                    gn_solver_->d_pixel_rela_weight_vec_,
                    gn_solver_->d_node_vec_, gn_solver_->img_rows_,
                    gn_solver_->img_cols_, gn_solver_->node_num_, weight_);

    // 确保CUDA计算任务已经完成
    checkCudaErrors(cudaDeviceSynchronize());
    // 检查CUDA执行过程中是否有错误发生
    checkCudaErrors(cudaGetLastError());
}

/**
 * 初始化平滑项约束
 *
 * 本函数用于初始化平滑项约束，将平滑项约束与高斯牛顿求解器关联，并设置约束的权重。
 * 这个约束主要用于优化过程中，以平滑目标函数中各变量的变化。
 *
 * @param gn_solver 指向GNSolver对象的指针，用于获取求解器的相关参数和状态。
 * @param weight 平滑项的权重，用于控制平滑程度。
 * @return 总是返回true，表示初始化成功。
 */
bool SmoothTermConstraint::Init(GNSolver* gn_solver, float weight) {
    assert(gn_solver);   // 确保gn_solver不为nullptr

    gn_solver_ = gn_solver;                 // 存储传入的GNSolver指针
    row_ = gn_solver_->triangle_num_ * 2;   // 根据三角形数量计算行数
    col_ = gn_solver_->vars_num_;           // 根据变量数量设置列数
    d_b_.resize(row_);   // 根据行数调整d_b_向量的大小
    SetWeight(weight);   // 设置平滑项的权重

    return true;
}

bool SmoothTermConstraint::Init() {
    row_ = gn_solver_->triangle_num_ * 2;
    col_ = gn_solver_->vars_num_;
    d_b_.resize(row_);

    return true;
}

/**
 * 计算平滑项约束的JTJ矩阵和JTb向量，并将结果存储到指定的设备内存中。
 * @param:
 *    - d_JTJ_a: 指向存储JTJ矩阵系数的设备内存的指针。
 *    - d_JTJ_ia: 指向存储JTJ矩阵索引的设备内存的指针，用于稀疏矩阵存储。
 *    - d_JTb: 指向存储JTb向量的设备内存的指针。
 *    - d_x: 指向当前迭代解的设备内存的指针，作为计算的输入。
 * @return: 无
 */

void SmoothTermConstraint::GetJTJAndJTb(float* d_JTJ_a, int* d_JTJ_ia,
                                        float* d_JTb, float* d_x) {
    // 应用平滑项约束的b函数，更新当前迭代解。
    b(d_x);
    // 计算并存储平滑项约束的JTJ矩阵系数和索引到设备内存。
    DirectiveJTJ(d_JTJ_a, d_JTJ_ia);
    // 计算并存储平滑项约束的JTb向量到设备内存。
    DirectiveJTb(d_JTb);
}

/**
 * 在GPU上计算平滑项残差的内核函数。
 *
 * @param b 存储残差结果的浮点数数组
 * @param triangle_num 三角形的数量
 * @param node_vec 存储节点向量的浮点数数组
 * @param node_rela_idx_vec 存储与三角形相关的节点索引的整数数组
 * @param triangle_u 三角形的u参数
 * @param triangle_v 三角形的v参数
 * @param weight 权重因子
 *
 * 该函数通过块线程的方式，对每个三角形计算平滑项的残差，并将结果存储到数组b中。
 */
__global__ void CalcSmoothTermResidualKernel(float* b, int triangle_num,
                                             float2* node_vec,
                                             int* node_rela_idx_vec,
                                             float triangle_u, float triangle_v,
                                             float weight) {
    // 计算当前线程处理的三角形索引
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // 超出三角形数量范围的线程直接返回
    if (idx >= triangle_num)
        return;

    // 获取三角形的三个节点向量
    float2 v1 = node_vec[node_rela_idx_vec[idx * 3]];
    float2 v2 = node_vec[node_rela_idx_vec[idx * 3 + 1]];
    float2 v3 = node_vec[node_rela_idx_vec[idx * 3 + 2]];

    // 计算并存储平滑项残差
    b[2 * idx]     = weight * (v1.x - (v2.x + triangle_u * (v3.x - v2.x) +
                                   triangle_v * (v3.y - v2.y)));
    b[2 * idx + 1] = weight * (v1.y - (v2.y + triangle_u * (v3.y - v2.y) +
                                       triangle_v * (-v3.x + v2.x)));
}

/**
 * 为SmoothTermConstraint类的成员函数b提供实现，该函数用于计算平滑项约束的右端项。
 *
 * @param d_x
 * 指向设备内存中某个数组的指针，该参数在函数内未使用，但可能用于后续扩展。
 */
void SmoothTermConstraint::b(float* d_x) {
    // 在设备内存上初始化右端项数组d_b_为0
    checkCudaErrors(cudaMemset(RAW_PTR(d_b_), 0,
                               gn_solver_->triangle_num_ * 2 * sizeof(float)));

    // 计算执行kernel函数CalcSmoothTermResidualKernel的网格和块的大小
    int block = 128, grid = (gn_solver_->triangle_num_ + block - 1) / block;
    // 调用CUDA kernel函数计算平滑项残差，并将结果存储到d_b_中
    CalcSmoothTermResidualKernel<<<grid, block>>>(
        RAW_PTR(d_b_), gn_solver_->triangle_num_,
        RAW_PTR(gn_solver_->d_node_vec_),
        RAW_PTR(gn_solver_->d_node_rela_idx_vec_), gn_solver_->triangle_u_,
        gn_solver_->triangle_v_, weight_);
    // 确保所有CUDA操作都已完成，检查有无错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 计算平滑项的JTJ矩阵的指令集。
 * 该函数用于通过CUDA实现对平滑项的JTJ矩阵的计算，借助于GPU的并行计算能力以提升计算效率。
 *
 * @param d_JTJ_a 指向存储JTJ矩阵系数的设备内存的指针。
 * @param d_JTJ_ia 指向存储JTJ矩阵索引的设备内存的指针。
 *
 * 该函数不返回任何值，但会将计算结果存储到指定的设备内存中。
 */
void SmoothTermConstraint::DirectiveJTJ(float* d_JTJ_a, int* d_JTJ_ia) {
    // 计算平滑项的JTJ矩阵
    CalcSmoothTermJTJ(d_JTJ_a, d_JTJ_ia, *gn_solver_->nz_blocks_static_,
                      gn_solver_->d_node_vec_, gn_solver_->node_num_,
                      gn_solver_->pixel_num_, gn_solver_->d_node_rela_idx_vec_,
                      gn_solver_->triangle_u_, gn_solver_->triangle_v_,
                      weight_);
    // 确保所有CUDA操作都已完成，避免潜在的异步问题
    checkCudaErrors(cudaDeviceSynchronize());
    // 检查是否有CUDA错误发生
    checkCudaErrors(cudaGetLastError());
}

/**
 * 计算平滑项约束的导数 JTb
 *
 * 该函数负责计算平滑项约束的导数 JTb，并将结果存储在传入的 d_JTb 数组中。
 * 这是通过调用 CalcSmoothTermJTb 函数实现的，该函数具体执行计算的过程。
 *
 * @param d_JTb 指向 CUDA 设备内存中用于存储计算结果的浮点数数组的指针。
 *              计算结果为平滑项约束的导数 JTb。
 * 注意：该函数不返回任何值，但会通过传入的参数修改数组内容。
 */
void SmoothTermConstraint::DirectiveJTb(float* d_JTb) {
    // 计算平滑项的导数 JTb，并将结果存储在 d_JTb 中
    CalcSmoothTermJTb(d_JTb, RAW_PTR(d_b_), *gn_solver_->nz_blocks_static_,
                      gn_solver_->d_node_vec_, gn_solver_->node_num_,
                      gn_solver_->pixel_num_, gn_solver_->d_node_rela_idx_vec_,
                      gn_solver_->triangle_u_, gn_solver_->triangle_v_,
                      weight_);

    // 确保 CUDA 操作完成并检查是否有错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 初始化零项约束
 * 该函数用于初始化零项约束，将约束应用于梯度下降法（GNSolver）中，确保某些变量的更新为零。
 *
 * @param gn_solver 指向GNSolver对象的指针，GNSolver是进行梯度下降解算的主对象。
 * @param weight 权重参数，用于调整零项约束的影响力。
 * @return 总是返回true，表示初始化成功。
 */
bool ZeroTermConstraint::Init(GNSolver* gn_solver, float weight) {
    assert(gn_solver);        // 确保gn_solver不为nullptr。
    gn_solver_ = gn_solver;   // 保存GNSolver指针到成员变量。
    row_       = gn_solver_->node_num_ * 2;   // 根据节点数量计算约束矩阵的行数。
    col_ = gn_solver_->vars_num_;   // 根据变量数量计算约束矩阵的列数。
    d_b_.resize(row_);   // 调整约束偏置项d_b_的大小，以适应行数。
    SetWeight(weight);   // 设置约束的权重。

    return true;
}

bool ZeroTermConstraint::Init() {
    row_ = gn_solver_->node_num_ * 2;
    col_ = gn_solver_->vars_num_;
    d_b_.resize(row_);

    return true;
}

/**
 * 为ZeroTermConstraint类中的GetJTJAndJTb函数生成注释：
 * 此函数用于计算并获取雅可比矩阵(JT)的乘积(JTJ)和平行向量(JTb)，
 * 以及更新输入向量x的值。
 *
 * @param d_JTJ_a 指向浮点数数组的指针，用于存储雅可比矩阵JTJ的数值部分。
 * @param d_JTJ_ia 指向整数数组的指针，用于存储雅可比矩阵JTJ的索引部分。
 * @param d_JTb
 * 指向浮点数数组的指针，用于存储雅可比矩阵JT的转置乘以向量b的结果。
 * @param d_x
 * 指向浮点数数组的指针，输入为当前的向量x，函数执行过程中可能会被更新。
 */
void ZeroTermConstraint::GetJTJAndJTb(float* d_JTJ_a, int* d_JTJ_ia,
                                      float* d_JTb, float* d_x) {
    // 调用b函数更新d_x指向的向量
    b(d_x);
    // 调用DirectiveJTJ函数计算并存储雅可比矩阵JTJ的数值和索引
    DirectiveJTJ(d_JTJ_a, d_JTJ_ia);
    // 调用DirectiveJTb函数计算并存储雅可比矩阵JT的转置乘以向量b的结果
    DirectiveJTb(d_JTb);
}

__global__ void CalcZeroTermResidualKernel(float* JTb, float* b,
                                           float2* original_node_vec,
                                           float2* node_vec, int node_num,
                                           float weight) {
    int node_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_idx >= node_num)
        return;

    float2 res = weight * (node_vec[node_idx] - original_node_vec[node_idx]);

    b[2 * node_idx]     = res.x;
    b[2 * node_idx + 1] = res.y;

    JTb[2 * node_idx] -= weight * res.x;
    JTb[2 * node_idx + 1] -= weight * res.y;
}

void ZeroTermConstraint::b(float* d_x) { }

/**
 * 计算零点约束的JTJ矩阵项
 *
 * @param d_JTJ_a 指向CUDA设备上JTJ矩阵数据的指针
 * @param d_JTJ_ia 指向CUDA设备上JTJ矩阵索引的指针
 * @param nz_blocks_static_ 静态非零块的数量
 * @param d_original_node_vec_ 原始节点向量的设备内存指针
 * @param d_node_vec_ 节点向量的设备内存指针
 * @param node_num_ 节点数量
 * @param weight_ 权重系数
 */
void ZeroTermConstraint::DirectiveJTJ(float* d_JTJ_a, int* d_JTJ_ia) {
    // 计算零点约束的JTJ矩阵
    CalcZeroTermJTJ(d_JTJ_a, d_JTJ_ia, *gn_solver_->nz_blocks_static_,
                    gn_solver_->d_original_node_vec_, gn_solver_->d_node_vec_,
                    gn_solver_->node_num_, weight_);
    // 确保CUDA操作完成
    checkCudaErrors(cudaDeviceSynchronize());
    // 检查CUDA是否发生错误
    checkCudaErrors(cudaGetLastError());
}

/**
 * 计算零点约束的JTb向量项
 *
 * @param d_JTb 指向CUDA设备上JTb向量的指针
 */
void ZeroTermConstraint::DirectiveJTb(float* d_JTb) {
    // 设置CUDA kernel的块和网格大小
    int block = 256, grid = (gn_solver_->node_num_ + block - 1) / block;
    // 调用CUDA kernel计算零点约束的残差
    CalcZeroTermResidualKernel<<<grid, block>>>(
        d_JTb, RAW_PTR(d_b_), gn_solver_->d_original_node_vec_,
        gn_solver_->d_node_vec_, gn_solver_->node_num_, weight_);
    // 确保CUDA操作完成
    checkCudaErrors(cudaDeviceSynchronize());
    // 检查CUDA是否发生错误
    checkCudaErrors(cudaGetLastError());
}
