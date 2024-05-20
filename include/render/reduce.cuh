#ifndef GALAHOLOSYSTEM_REDUCE_H
#define GALAHOLOSYSTEM_REDUCE_H

#include "nz_block_statistics.cuh"
#include "util/helper_cuda.h"
#include "util/math_utils.h"
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>

/**
 * 计算双线性插值的系数
 *
 * 该函数用于根据给定的坐标和区间边界计算双线性插值所需的四个系数。
 * 这些系数可用于从四个角点的值插值出任意点的值。
 *
 * @param coef_00 双线性插值系数00（对应左上角点）
 * @param coef_10 双线性插值系数10（对应右上角点）
 * @param coef_01 双线性插值系数01（对应左下角点）
 * @param coef_11 双线性插值系数11（对应右下角点）
 * @param x 待插值点的x坐标
 * @param y 待插值点的y坐标
 * @param x_0 x坐标的左边界
 * @param y_0 y坐标的上边界
 * @param x_1 x坐标的右边界
 * @param y_1 y坐标的下边界
 */
__device__ __host__ __forceinline__ void
CalcBilinearCoefs(float& coef_00, float& coef_10, float& coef_01,
                  float& coef_11, float x, float y, float x_0, float y_0,
                  float x_1, float y_1) {
    // 计算x和y方向的归一化系数
    float norm_coef_x = 1 / (x_1 - x_0);
    float norm_coef_y = 1 / (y_1 - y_0);

    // 计算x和y方向的插值系数
    float coef_x_0 = (x_1 - x) * norm_coef_x;
    float coef_y_0 = (y_1 - y) * norm_coef_y;

    // 计算四个双线性插值系数
    coef_00 = coef_x_0 * coef_y_0;
    coef_10 = (1 - coef_x_0) * coef_y_0;
    coef_01 = coef_x_0 * (1 - coef_y_0);
    coef_11 = (1 - coef_x_0) * (1 - coef_y_0);
}

/**
 * 在给定的图像上，使用双线性插值计算指定点的梯度值。
 *
 * @param img
 * 指向图像数据的cuda::PtrStepSz<float>对象，包含图像的宽度、高度和数据。
 * @param x 想要计算梯度的点的x坐标（浮点数）。
 * @param y 想要计算梯度的点的y坐标（浮点数）。
 * @return 计算得到的梯度值。如果坐标超出图像边界或像素值异常，则返回0.0f。
 */
__device__ __forceinline__ float
GetGradientValueBilinearDevice(cv::cuda::PtrStepSz<float> img, float x,
                               float y) {
    // 计算四个最近的整数点坐标
    int x_0 = (int)x;
    int y_0 = (int)y;
    int x_1 = x_0 + 1;
    int y_1 = y_0 + 1;

    // 检查坐标是否在图像范围内
    if (x_0 < 0 || y_0 < 0 || x_1 >= img.cols || y_1 >= img.rows)
        return 0.0f;

    // 计算插值系数
    float coef_x_0 = x_1 - x;
    float coef_y_0 = y_1 - y;

    float coef_00 = coef_x_0 * coef_y_0;
    float coef_10 = (1 - coef_x_0) * coef_y_0;
    float coef_01 = coef_x_0 * (1 - coef_y_0);
    float coef_11 = (1 - coef_x_0) * (1 - coef_y_0);

    // 获取四个点的值，并检查值是否异常
    float val_00 = img.ptr(y_0)[x_0];
    if (val_00 > 2.0 || val_00 < -2.0)   // 梯度有正有负，此处假设值异常
        return 0.0f;
    float val_10 = img.ptr(y_0)[x_1];
    if (val_10 > 2.0 || val_10 < -2.0)
        return 0.0f;
    float val_01 = img.ptr(y_1)[x_0];
    if (val_01 > 2.0 || val_01 < -2.0)
        return 0.0f;
    float val_11 = img.ptr(y_1)[x_1];
    if (val_11 > 2.0 || val_11 < -2.0)
        return 0.0f;

    // 使用双线性插值计算最终的梯度值
    return coef_00 * val_00 + coef_01 * val_01 + coef_10 * val_10 +
           coef_11 * val_11;
}

/**
 * 在给定的图像上，通过双线性插值计算指定像素点的值。
 *
 * @param img 指向cuda::Mat对象的指针，包含要处理的图像数据。
 * @param x 想要查询像素的x坐标（浮点数）。
 * @param y 想要查询像素的y坐标（浮点数）。
 * @return
 * 计算得到的像素值，如果坐标超出图像边界或像素值大于2.0，则返回65535.0f。
 */
__device__ __forceinline__ float
GetPixelValueBilinearDevice(cv::cuda::PtrStepSz<float> img, float x, float y) {
    // 计算像素点的整数部分和其右边、上边的像素点位置
    int x_0 = (int)x;
    int y_0 = (int)y;
    int x_1 = x_0 + 1;
    int y_1 = y_0 + 1;

    // 检查计算出的像素位置是否在图像范围内
    if (x_0 < 0 || y_0 < 0 || x_1 >= img.cols || y_1 >= img.rows)
        return 65535.0f;

    // 计算插值系数
    float coef_x_0 = x_1 - x;
    float coef_y_0 = y_1 - y;

    float coef_00 = coef_x_0 * coef_y_0;
    float coef_10 = (1 - coef_x_0) * coef_y_0;
    float coef_01 = coef_x_0 * (1 - coef_y_0);
    float coef_11 = (1 - coef_x_0) * (1 - coef_y_0);

    // 获取四个相邻像素的值，并检查是否超出阈值
    float val_00 = img.ptr(y_0)[x_0];
    if (val_00 > 2.0)
        return 65535.0f;
    float val_10 = img.ptr(y_0)[x_1];
    if (val_10 > 2.0)
        return 65535.0f;
    float val_01 = img.ptr(y_1)[x_0];
    if (val_01 > 2.0)
        return 65535.0f;
    float val_11 = img.ptr(y_1)[x_1];
    if (val_11 > 2.0)
        return 65535.0f;

    // 根据插值系数计算最终像素值
    return coef_00 * val_00 + coef_01 * val_01 + coef_10 * val_10 +
           coef_11 * val_11;
}

inline __host__ __device__ float3 operator*(float a, uchar4& b) {
    float3 res;
    res.x = (int)b.x * a;
    res.y = (int)b.y * a;
    res.z = (int)b.z * a;

    return res;
}

void CalcDataTermJTJ(float* d_JTJ_a, int* d_JTJ_ia, cv::cuda::GpuMat& d_dx_img,
                     cv::cuda::GpuMat& d_dy_img, NzBlockStatisticsForJTJ& Iij,
                     float2* d_warped_pixal_vec, int* d_pixel_rela_idx_vec,
                     float* d_pixel_rela_weight_vec, float2* d_node_vec,
                     int rows, int cols, int node_num, float weight);

void CalcDataTermJTb(float* d_JTb, float* d_b, cv::cuda::GpuMat& d_dx_img,
                     cv::cuda::GpuMat& d_dy_img,
                     NzBlockStatisticsForJTJ& nz_blocks_static,
                     float2* d_warped_pixal_vec, int* d_pixel_rela_idx_vec,
                     float* d_pixel_rela_weight_vec, float2* d_node_vec,
                     int rows, int cols, int node_num, float weight);

void CalcSmoothTermJTJ(float* d_JTJ_a, int* d_JTJ_ia,
                       NzBlockStatisticsForJTJ& Iij, float2* d_node_vec,
                       int node_num, int pixel_num, int* d_node_rela_idx_vec_,
                       float triangle_u, float triangle_v, float weight);

void CalcSmoothTermJTb(float* d_JTb, float* d_b, NzBlockStatisticsForJTJ& Iij,
                       float2* d_node_vec, int node_num, int pixel_num,
                       int* d_node_rela_idx_vec_, float triangle_u,
                       float triangle_v, float weight);

void CalcZeroTermJTJ(float* d_JTJ_a, int* d_JTJ_ia,
                     NzBlockStatisticsForJTJ& Iij, float2* d_original_node_vec,
                     float2* d_node_vec, int node_num, float weight);

#endif   // GALAHOLOSYSTEM_REDUCE_H
