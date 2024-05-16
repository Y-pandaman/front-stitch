//
// Created by vradmin on 17-11-2.
//

#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#ifdef __CLION_IDE__
#define __device__
#define __host__
#define __global__
#define __CUDACC__
#endif

#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include <cuda_runtime.h>

/**
 * 将 thrust 设备向量转换为原始指针。
 *
 * @param thrust_device_vector 一个 thrust 的设备向量，其类型为
 * `thrust::device_vector<T>`。
 * @return 返回一个指向设备向量数据起始位置的原始指针。
 */
#define RAW_PTR(thrust_device_vector) \
    thrust::raw_pointer_cast(&thrust_device_vector[0])

/**
 * 将原始设备指针转换为 thrust 迭代器。
 *
 * @param raw_device_ptr 一个指向设备内存中的数据的原始指针。
 * @return 返回一个对应的 thrust 设备迭代器。
 */
#define THRUST_ITER(raw_device_ptr) thrust::device_pointer_cast(raw_device_ptr)

#define GRID_SIZE(total_num, block_size) \
    (total_num + block_size - 1) / block_size

#define CUDAMALLOC(dst, length, type) \
    checkCudaErrors(cudaMalloc((void**)&dst, sizeof(type) * length))
#define CUDAFREE(dst) checkCudaErrors(cudaFree(dst))
#define UPLOAD(d_dst, h_src, length, type)                          \
    checkCudaErrors(cudaMemcpy(d_dst, h_src, sizeof(type) * length, \
                               cudaMemcpyHostToDevice))
#define DOWNLOAD(h_dst, d_src, length, type)                        \
    checkCudaErrors(cudaMemcpy(h_dst, d_src, sizeof(type) * length, \
                               cudaMemcpyDeviceToHost))

#endif   //_CUDA_UTILS_H
