#ifndef __RENDER__
#define __RENDER__

#include "cuda_runtime.h"
#include "cylinder_stitcher.cuh"
#include "project_to_cylinder.cuh"
#include <cuda.h>
#include <iostream>
#include <thrust/extrema.h>
#include <vector>

static inline __device__ void get_intrinsic(float& fx, float& fy, float& cx,
                                            float& cy, float image_theta_range,
                                            float image_phi_range, int height,
                                            int width) {
    fx = (width / 2.0f) / tanf(image_theta_range / 2.0f);
    fy = height / image_phi_range;
    cx = (width - 1.0f) / 2.0f;
    cy = (height - 1.0f) / 2.0f;
}

/**
 * @brief 使用给定的逆矩阵旋转向量
 * 
 * 通过矩阵乘法，使用给定的三个向量（X轴，Y轴，Z轴）作为矩阵的列，对输入的向量v进行旋转。
 * 这个函数假设这三个向量是构成一个正交矩阵的列，且它们已经归一化。
 * 
 * @param X_axis X轴向量，作为矩阵的的第一列
 * @param Y_axis Y轴向量，作为矩阵的的第二列
 * @param Z_axis Z轴向量，作为矩阵的的第三列
 * @param v 待旋转的向量
 * @return float3 旋转后的向量
 */
static inline __device__ float3 rotateVector_inv(float3 X_axis, float3 Y_axis,
                                                 float3 Z_axis, float3 v) {
    // 通过矩阵乘法计算旋转后的向量，其中X_axis, Y_axis, Z_axis 视为矩阵的列
    return make_float3(X_axis.x * v.x + Y_axis.x * v.y + Z_axis.x * v.z,
                       X_axis.y * v.x + Y_axis.y * v.y + Z_axis.y * v.z,
                       X_axis.z * v.x + Y_axis.z * v.y + Z_axis.z * v.z);
}

/**
 * @brief 通过给定的三个轴向量对另一个向量进行旋转
 *
 * @param X_axis X轴的单位向量
 * @param Y_axis Y轴的单位向量
 * @param Z_axis Z轴的单位向量
 * @param v 需要被旋转的向量
 * @return float3 旋转后的向量
 */
static inline __device__ float3 rotateVector(float3 X_axis, float3 Y_axis,
                                             float3 Z_axis, float3 v) {
    // 根据旋转矩阵的公式，计算旋转后的向量
    return make_float3(X_axis.x * v.x + X_axis.y * v.y + X_axis.z * v.z,
                       Y_axis.x * v.x + Y_axis.y * v.y + Y_axis.z * v.z,
                       Z_axis.x * v.x + Z_axis.y * v.y + Z_axis.z * v.z);
}

__host__ void BlendExtraViewToScreen_cuda(uchar3* dst_cyl_img,
                                          uchar3* src_cyl_img, int width,
                                          int height, float w);

__host__ void BlendExtraViewToScreen4Channels_cuda(uchar3* dst_cyl_img,
                                                   uchar4* src_cyl_img,
                                                   int width, int height,
                                                   float w);

__host__ void RenderToScreen_cuda(std::vector<float4*>& novel_view_intrins,
                                  std::vector<float3*>& novel_view_extrin_Rs,
                                  CylinderImageGPU cyl_img,
                                  std::vector<uchar3*> novel_images,
                                  CylinderGPU cylinder, float3 origin,
                                  int novel_images_num);

#endif