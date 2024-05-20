#include "calc/reduce.cuh"
#include "util/cuda_utils.h"
#include "util/helper_cuda.h"
#include "util/math_utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_functions.h>
#include <vector_types.h>

/**
 * 在 warp 内部进行求和减少操作。
 * 对于一个 float4 类型的值，此函数将在同一个 warp 的所有线程之间进行同步，并对
 * x、y、z、w 分量进行求和。
 *
 * @param val float4 类型的输入值，包含 x、y、z、w 四个分量。
 * @return 经过在 warp 内部求和减少后的 float4 类型值。
 */
__inline__ __device__ float4 WarpReduceSum(float4 val) {
    // 遍历 warp 中的线程，逐步将值减少到 single thread
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        // 使用 __shfl_down_sync 函数将值从下一位线程中获取并相加
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, offset);
        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, offset);
        val.z += __shfl_down_sync(0xFFFFFFFF, val.z, offset);
        val.w += __shfl_down_sync(0xFFFFFFFF, val.w, offset);
    }

    return val;
}

/**
 * 在CUDA中，对一个线程块中的数据进行求和减少操作。
 * 这个函数主要用于在GPU的线程块级别对float4类型的值进行缩减操作，例如求和，
 * 以便于进一步在块级别进行聚合。
 *
 * @param val 每个线程传入的float4类型的值，该值将被缩减。
 * @return 返回线程块中所有线程的float4值的缩减结果（例如求和）。
 */
__inline__ __device__ float4 BlockReduceSum(float4 val) {
    // 静态共享内存用于存储每个warp的缩减结果
    static __shared__ float4 shared[32];
    // 计算当前线程所在的lane和warp ID
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    // 使用WarpReduceSum函数对val进行warp级别的减少操作
    val = WarpReduceSum(val);

    // 将warp级别的减少结果写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();   // 确保所有线程都完成了写入操作

    const float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
    // 从共享内存读取warp的减少结果，如果当前warp不存在，则使用零
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    // 如果当前线程在第一个warp中，则进一步对这个warp的结果进行减少操作
    if (wid == 0) {
        val = WarpReduceSum(val);
    }

    return val;   // 返回线程块的最终减少结果
}

__global__ void ReduceSum(float4* out, float4* in, int N) {
    float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        sum += in[i];
    }

    sum = BlockReduceSum(sum);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

/**
 * 在 warp 内部进行求和减少操作。
 * 对于给定的 float2（x, y）值，它将该值与 warp
 * 中处于较低位置的对应元素进行累加， 从而实现 warp 内部的求和。该操作依赖于
 * CUDA 的同步共享内存指令。
 *
 * @param val 一个 float2 结构体，包含需要进行累加的两个浮点数值。
 * @return 返回累加后的 float2 结构体，其中的 x 和 y 分别为原始值与 warp
 * 中对应位置元素累加后的结果。
 */
__inline__ __device__ float2 WarpReduceSum(float2 val) {
    // 遍历 warp 中的所有线程，将值沿 warp 下降方向累加
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        // 沿着 warp 的下降方向，将值从右边的线程累加到当前线程的 x 和 y 上
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, offset);
        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, offset);
    }

    return val;
}

/**
 * 在CUDA线程块内对浮点数对（float2）进行求和减少操作。
 * 这个函数首先在 warp（单个线程块内的轻量级线程组）级别上进行求和减少，
 * 然后对来自所有 warp 的结果进行进一步的求和减少，以得到整个线程块的减少结果。
 *
 * @param val 每个线程传入的初始值，类型为float2（两个浮点数的组合）。
 * @return 返回线程块级别的浮点数对求和减少结果。
 */
__inline__ __device__ float2 BlockReduceSum(float2 val) {
    // 静态共享内存用于存储每个warp的减少结果
    static __shared__ float2 shared[32];
    // 计算当前线程所在的lane和warp ID
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    // 在warp内进行求和减少
    val = WarpReduceSum(val);

    // 将warp的减少结果写入共享内存
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();   // 确保所有线程都完成写入

    const float2 zero = {0.0f, 0.0f};
    // 从共享内存读取warp的减少结果，如果该warp不存在，则使用0
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    // 如果是第一个warp，则对其结果进行再次减少
    if (wid == 0) {
        val = WarpReduceSum(val);
    }

    return val;
}

__global__ void ReduceSum(float2* out, float2* in, int N) {
    float2 sum = {0.0f, 0.0f};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        sum += in[i];
    }

    sum = BlockReduceSum(sum);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

/**
 * 在给定的整型数组中查找指定值的索引。
 *
 * @param node_list 整型数组，待搜索的目标数组。
 * @param val 要搜索的整数值。
 * @param num 数组的元素数量。
 * @return 找到的元素的索引，如果未找到则返回-1。
 */
__forceinline__ __device__ int FindIndex(int* node_list, int val, int num) {
    // 遍历数组，搜索指定的值
    for (int i = 0; i < num; i++) {
        if (node_list[i] == val) {
            return i;   // 找到值时，返回其索引
        }
    }
    return -1;   // 未找到值时，返回-1
}

struct DataTermJTJReduction {
    float* JTJ_a_;

    int* JTJ_ia_;
    cv::cuda::PtrStepSz<float> dx_img_;
    cv::cuda::PtrStepSz<float> dy_img_;
    int* nz_block_idx_vec_;
    int* function_idx_vec_;
    int* offset_vec_;
    int* data_term_function_num_vec_;
    int* pre_nz_block_num_vec_;
    float2* warped_pixal_vec_;
    int* pixel_rela_idx_vec_;
    float* pixel_rela_weight_vec_;
    float2* node_vec_;
    int rows_;
    int cols_;
    int node_num_;
    float weight_;

    /**
     * 在设备上获取两个节点针对指定像素的导数乘积。
     *
     * @param node_idx_i 第一个节点的索引。
     * @param node_idx_j 第二个节点的索引。
     * @param pixel_idx 指定的像素索引。
     * @return 包含四个导数乘积的float4向量：(deri_i_x * deri_j_x, deri_i_x *
     * deri_j_y, deri_i_y * deri_j_x, deri_i_y * deri_j_y)。
     */
    __device__ __forceinline__ float4 GetProducts(int node_idx_i,
                                                  int node_idx_j,
                                                  int pixel_idx) {
        // 获取指定像素的-warped-像素坐标。
        float2 warped_pixel = warped_pixal_vec_[pixel_idx];
        // 如果warped像素坐标有任何一个小于0，则直接返回所有分量为0的float4。
        if (warped_pixel.x < 0.0f || warped_pixel.y < 0.0f) {
            return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // 使用双线性插值获取指定warped像素在dy_img_和dx_img_上的梯度值。
        float dy_val = GetGradientValueBilinearDevice(dy_img_, warped_pixel.x,
                                                      warped_pixel.y);
        float dx_val = GetGradientValueBilinearDevice(dx_img_, warped_pixel.x,
                                                      warped_pixel.y);

        int node_offset;
        float weight_P2N;
        // 计算第一个节点针对指定像素的加权导数。
        node_offset =
            FindIndex(pixel_rela_idx_vec_ + pixel_idx * 4, node_idx_i, 4);
        weight_P2N =
            weight_ * pixel_rela_weight_vec_[pixel_idx * 4 + node_offset];
        float deri_i_y = weight_P2N * dy_val;
        float deri_i_x = weight_P2N * dx_val;

        // 计算第二个节点针对指定像素的加权导数。
        node_offset =
            FindIndex(pixel_rela_idx_vec_ + pixel_idx * 4, node_idx_j, 4);
        weight_P2N =
            weight_ * pixel_rela_weight_vec_[pixel_idx * 4 + node_offset];
        float deri_j_y = weight_P2N * dy_val;
        float deri_j_x = weight_P2N * dx_val;

        // 返回两个节点的导数乘积。
        return make_float4(deri_i_x * deri_j_x, deri_i_x * deri_j_y,
                           deri_i_y * deri_j_x, deri_i_y * deri_j_y);
    }

    /**
     * 在CUDA设备上执行的函数，用于计算并累加节点间函数的乘积，进而更新JTJ矩阵。
     * 该函数不接受参数，也不返回值，通过共享内存和块级归约来实现并行计算和结果的合并。
     */
    __device__ __forceinline__ void operator()() {
        // 初始化累加和
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

        // 根据块索引计算非零块索引、节点索引i和j以及函数数量
        int nz_block_idx    = nz_block_idx_vec_[blockIdx.x];
        int node_idx_i      = nz_block_idx / node_num_;
        int node_idx_j      = nz_block_idx % node_num_;
        int function_num    = data_term_function_num_vec_[blockIdx.x];
        int* p_function_idx = function_idx_vec_ + offset_vec_[blockIdx.x];

        // 使用线程块级循环计算所有函数的乘积并累加到sum中
        for (int i = threadIdx.x; i < function_num; i += blockDim.x) {
            float4 val = GetProducts(node_idx_i, node_idx_j, p_function_idx[i]);

            sum += val;
        }

        // 使用块级归约技术将线程块内的累加和合并为一个值
        sum = BlockReduceSum(sum);

        // 只有线程索引为0的线程才更新JTJ矩阵
        if (threadIdx.x == 0) {
            // 如果节点i等于节点j，则更新对应的对角线元素
            if (node_idx_i == node_idx_j) {
                int start_pos =
                    pre_nz_block_num_vec_[node_idx_i * node_num_ + node_idx_j] *
                    2;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 0] + start_pos + 0] += sum.x;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 0] + start_pos + 1] += sum.y;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 1] + start_pos + 0] += sum.z;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 1] + start_pos + 1] += sum.w;
            } else {
                // 如果节点i不等于节点j，则分别更新两个节点对应的非对角线元素
                int start_pos =
                    pre_nz_block_num_vec_[node_idx_i * node_num_ + node_idx_j] *
                    2;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 0] + start_pos + 0] += sum.x;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 0] + start_pos + 1] += sum.y;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 1] + start_pos + 0] += sum.z;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 1] + start_pos + 1] += sum.w;

                start_pos =
                    pre_nz_block_num_vec_[node_idx_j * node_num_ + node_idx_i] *
                    2;
                JTJ_a_[JTJ_ia_[node_idx_j * 2 + 0] + start_pos + 0] += sum.x;
                JTJ_a_[JTJ_ia_[node_idx_j * 2 + 0] + start_pos + 1] += sum.z;
                JTJ_a_[JTJ_ia_[node_idx_j * 2 + 1] + start_pos + 0] += sum.y;
                JTJ_a_[JTJ_ia_[node_idx_j * 2 + 1] + start_pos + 1] += sum.w;
            }
        }
    }
};
/**
 * 在GPU上执行数据项JTJ缩减的计算 kernel 函数。
 *
 * @param dtjtjr
 * 一个封装了数据项JTJ计算和缩减逻辑的对象。此对象应在调用此kernel函数前正确初始化。
 *
 * 该函数通过调用传入的对象的成员函数，执行数据项JTJ的计算和缩减过程。这是并行化计算的一部分，
 * 旨在利用GPU的并行计算能力加速这一耗时操作。
 */
__global__ void DataTermJTJReductionKernel(DataTermJTJReduction dtjtjr) {
    // 调用封装了数据项JTJ计算和缩减逻辑的对象的函数
    dtjtjr();
}

/**
 * 计算数据项JTJ矩阵的函数。
 * 该函数用于在CUDA设备上并行计算非零块的JTJ矩阵项，是优化过程的一部分。
 *
 * @param d_JTJ_a 指向存储JTJ矩阵系数的浮点数组的指针。
 * @param d_JTJ_ia 指向存储JTJ矩阵索引的整型数组的指针。
 * @param d_dx_img CUDA的GpuMat对象，存储图像的x方向梯度。
 * @param d_dy_img CUDA的GpuMat对象，存储图像的y方向梯度。
 * @param nz_blocks_static 包含非零块统计信息的对象。
 * @param d_warped_pixal_vec 指向存储变形像素向量的浮点型数组的指针。
 * @param d_pixel_rela_idx_vec 指向存储像素相对索引向量的整型数组的指针。
 * @param d_pixel_rela_weight_vec 指向存储像素相对权重向量的浮点型数组的指针。
 * @param d_node_vec 指向存储节点向量的浮点型数组的指针。
 * @param rows 图像的行数。
 * @param cols 图像的列数。
 * @param node_num 节点的数量。
 * @param weight 权重参数，用于调整计算结果。
 */
void CalcDataTermJTJ(float* d_JTJ_a, int* d_JTJ_ia, cv::cuda::GpuMat& d_dx_img,
                     cv::cuda::GpuMat& d_dy_img,
                     NzBlockStatisticsForJTJ& nz_blocks_static,
                     float2* d_warped_pixal_vec, int* d_pixel_rela_idx_vec,
                     float* d_pixel_rela_weight_vec, float2* d_node_vec,
                     int rows, int cols, int node_num, float weight) {
    // 初始化DataTermJTJReduction结构体，用于传递数据和控制计算。
    DataTermJTJReduction dtjtjr;

    // 设置DataTermJTJReduction结构体的成员变量。
    // 这些变量指向CUDA内存中的数据，包括矩阵系数、索引、图像梯度、非零块统计信息等。
    dtjtjr.JTJ_a_            = d_JTJ_a;
    dtjtjr.JTJ_ia_           = d_JTJ_ia;
    dtjtjr.dx_img_           = d_dx_img;
    dtjtjr.dy_img_           = d_dy_img;
    dtjtjr.nz_block_idx_vec_ = RAW_PTR(nz_blocks_static.d_nz_block_idx_vec_);
    dtjtjr.function_idx_vec_ = RAW_PTR(nz_blocks_static.d_function_idx_vec_);
    dtjtjr.offset_vec_       = RAW_PTR(nz_blocks_static.d_offset_vec_);
    dtjtjr.data_term_function_num_vec_ =
        RAW_PTR(nz_blocks_static.d_pixel_idx_num_vec_);
    dtjtjr.pre_nz_block_num_vec_ =
        RAW_PTR(nz_blocks_static.d_pre_nz_block_num_vec_);
    dtjtjr.warped_pixal_vec_      = RAW_PTR(d_warped_pixal_vec);
    dtjtjr.pixel_rela_idx_vec_    = RAW_PTR(d_pixel_rela_idx_vec);
    dtjtjr.pixel_rela_weight_vec_ = RAW_PTR(d_pixel_rela_weight_vec);
    dtjtjr.node_vec_              = RAW_PTR(d_node_vec);
    dtjtjr.rows_                  = rows;
    dtjtjr.cols_                  = cols;
    dtjtjr.node_num_              = node_num;
    dtjtjr.weight_                = weight;

    // 设置CUDA kernel的块和网格大小，并启动kernel进行计算。
    // Kernel将并行处理所有非零块的JTJ矩阵项。
    int block = 256, grid = nz_blocks_static.nz_block_num_;
    DataTermJTJReductionKernel<<<grid, block>>>(dtjtjr);

    // 检查CUDA执行过程中是否有错误发生，并确保所有操作都已完成。
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

struct DataTermJTbReduction {
    float* JTb_;

    float* b_;
    cv::cuda::PtrStepSz<float> dx_img_;
    cv::cuda::PtrStepSz<float> dy_img_;
    int* nz_block_idx_vec_;
    int* function_idx_vec_;
    int* offset_vec_;
    int* data_term_function_num_vec_;
    int* pre_nz_block_num_vec_;
    float2* warped_pixal_vec_;
    int* pixel_rela_idx_vec_;
    float* pixel_rela_weight_vec_;
    float2* node_vec_;
    int rows_;
    int cols_;
    int node_num_;
    float weight_;

    /**
     * 在设备上获取像素的梯度乘积。
     *
     * @param node_idx_i 节点索引i。
     * @param pixel_idx 像素索引。
     * @return float2类型，包含两个分量，分别是x和y方向的梯度乘积。
     */
    __device__ __forceinline__ float2 GetProducts(int node_idx_i,
                                                  int pixel_idx) {
        // 获取经过变形的像素值
        float2 warped_pixel = warped_pixal_vec_[pixel_idx];
        // 如果像素坐标小于0，则返回(0, 0)
        if (warped_pixel.x < 0.0f || warped_pixel.y < 0.0f) {
            return make_float2(0.0f, 0.0f);
        }

        // 使用双线性插值获取dy和dx的梯度值
        float dy_val = GetGradientValueBilinearDevice(dy_img_, warped_pixel.x,
                                                      warped_pixel.y);
        float dx_val = GetGradientValueBilinearDevice(dx_img_, warped_pixel.x,
                                                      warped_pixel.y);

        int node_offset;
        float weight_P2N;
        // 计算像素到节点的索引偏移，并获取相应的权重
        node_offset =
            FindIndex(pixel_rela_idx_vec_ + pixel_idx * 4, node_idx_i, 4);
        weight_P2N =
            weight_ * pixel_rela_weight_vec_[pixel_idx * 4 + node_offset];
        // 计算节点的x和y方向的梯度
        float deri_i_x = weight_P2N * dx_val;
        float deri_i_y = weight_P2N * dy_val;

        // 返回像素的梯度乘积
        return make_float2(deri_i_x * b_[pixel_idx], deri_i_y * b_[pixel_idx]);
    }

    /**
     * 在CUDA设备上执行的函数，用于计算特定非零块索引对应的节点的雅可比矩阵项。
     * 该函数为线程块级别的并行化设计，每个线程块处理一个非零块。
     *
     * 无参数和返回值，但操作共享内存中的数组和全局内存中的数据。
     */
    __device__ __forceinline__ void operator()() {
        // 根据线程块索引计算当前处理的非零块索引，并分解为节点i和节点j的索引
        int nz_block_idx = nz_block_idx_vec_[blockIdx.x];
        int node_idx_i   = nz_block_idx / node_num_;
        int node_idx_j   = nz_block_idx % node_num_;
        // 如果节点i和节点j不相等，则不进行计算，直接返回
        if (node_idx_i != node_idx_j) {
            return;
        }

        // 初始化累加和
        float2 sum = {0.0f, 0.0f};

        // 获取当前非零块关联的函数数量和函数索引的起始位置
        int function_num    = data_term_function_num_vec_[blockIdx.x];
        int* p_function_idx = function_idx_vec_ + offset_vec_[blockIdx.x];

        // 使用线程块内循环来并行计算所有函数的贡献，并累加到sum中
        for (int i = threadIdx.x; i < function_num; i += blockDim.x) {
            float2 val = GetProducts(node_idx_i, p_function_idx[i]);

            sum += val;
        }

        // 使用BlockReduceSum函数对线程块内的sum进行求和
        sum = BlockReduceSum(sum);

        // 如果是线程块中的第一个线程，则更新雅可比矩阵的对应项
        if (threadIdx.x == 0) {
            JTb_[node_idx_i * 2] -= sum.x;
            JTb_[node_idx_i * 2 + 1] -= sum.y;
        }
    }
};
/**
 * 在GPU上执行数据项JTb的缩减操作的内核函数。
 *
 * __global__表示此函数可以在GPU的多个线程中并行执行。
 *
 * @param dtjtbr
 * 一个DataTermJTbReduction类型的对象，封装了执行数据项JTb缩减的具体操作。
 *               此对象作为一个函数调用，利用GPU线程并行执行。
 *
 * 该内核函数不返回任何值。
 */
__global__ void DataTermJTbReductionKernel(DataTermJTbReduction dtjtbr) {
    // 调用封装在dtjtbr对象中的函数，执行数据项JTb的缩减计算
    dtjtbr();
}

/**
 * 计算数据项JTb
 *
 * 该函数用于根据给定的图像梯度、非零块统计信息、变形像素向量等，计算数据项JTb。
 * 数据项JTb是优化过程中的一项重要组成部分。
 *
 * @param d_JTb 指向CUDA设备内存中存放计算结果JTb的浮点数数组。
 * @param d_b 指向CUDA设备内存中存放b的浮点数数组。
 * @param d_dx_img CUDA的GpuMat对象，存储图像的x方向梯度。
 * @param d_dy_img CUDA的GpuMat对象，存储图像的y方向梯度。
 * @param nz_blocks_static 非零块统计信息，包含非零块的相关参数。
 * @param d_warped_pixal_vec 指向CUDA设备内存中存放变形像素向量的float2数组。
 * @param d_pixel_rela_idx_vec
 * 指向CUDA设备内存中存放像素相关索引向量的整型数组。
 * @param d_pixel_rela_weight_vec
 * 指向CUDA设备内存中存放像素相关权重向量的浮点数数组。
 * @param d_node_vec 指向CUDA设备内存中存放节点向量的float2数组。
 * @param rows 图像的行数。
 * @param cols 图像的列数。
 * @param node_num 节点数量。
 * @param weight 权重参数。
 */
void CalcDataTermJTb(float* d_JTb, float* d_b, cv::cuda::GpuMat& d_dx_img,
                     cv::cuda::GpuMat& d_dy_img,
                     NzBlockStatisticsForJTJ& nz_blocks_static,
                     float2* d_warped_pixal_vec, int* d_pixel_rela_idx_vec,
                     float* d_pixel_rela_weight_vec, float2* d_node_vec,
                     int rows, int cols, int node_num, float weight) {
    // 初始化DataTermJTbReduction结构体用于传递数据和参数到内核
    DataTermJTbReduction dtjtbr;

    // 设置结构体中的参数
    dtjtbr.JTb_              = d_JTb;
    dtjtbr.b_                = d_b;
    dtjtbr.dx_img_           = d_dx_img;
    dtjtbr.dy_img_           = d_dy_img;
    dtjtbr.nz_block_idx_vec_ = RAW_PTR(nz_blocks_static.d_nz_block_idx_vec_);
    dtjtbr.function_idx_vec_ = RAW_PTR(nz_blocks_static.d_function_idx_vec_);
    dtjtbr.offset_vec_       = RAW_PTR(nz_blocks_static.d_offset_vec_);
    dtjtbr.data_term_function_num_vec_ =
        RAW_PTR(nz_blocks_static.d_pixel_idx_num_vec_);
    dtjtbr.pre_nz_block_num_vec_ =
        RAW_PTR(nz_blocks_static.d_pre_nz_block_num_vec_);
    dtjtbr.warped_pixal_vec_      = RAW_PTR(d_warped_pixal_vec);
    dtjtbr.pixel_rela_idx_vec_    = RAW_PTR(d_pixel_rela_idx_vec);
    dtjtbr.pixel_rela_weight_vec_ = RAW_PTR(d_pixel_rela_weight_vec);
    dtjtbr.node_vec_              = RAW_PTR(d_node_vec);
    dtjtbr.rows_                  = rows;
    dtjtbr.cols_                  = cols;
    dtjtbr.node_num_              = node_num;
    dtjtbr.weight_                = weight;

    // 设置CUDA内核执行的网格和块大小
    int block = 256, grid = nz_blocks_static.nz_block_num_;
    // 调用CUDA内核进行计算
    DataTermJTbReductionKernel<<<grid, block>>>(dtjtbr);
    // 检查CUDA执行过程中是否有错误发生
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

struct SmoothTermJTJReduction {
    float* JTJ_a_;

    int* JTJ_ia_;
    int* nz_block_idx_vec_;
    int* function_idx_vec_;
    int* offset_vec_;
    int* data_term_function_num_vec_;
    int* pre_nz_block_num_vec_;
    float2* node_vec_;
    int node_num_;
    int pixel_num_;
    int* node_rela_idx_vec_;
    float triangle_u_;
    float triangle_v_;
    float weight_;

    /**
     * @brief 计算两个节点在指定三角形内的导数乘积。
     *
     * @param node_idx_i 第一个节点的索引
     * @param node_idx_j 第二个节点的索引
     * @param triangle_idx 指定的三角形索引
     * @return float4 包含四个导数乘积的浮点数数组（dx_i * dx_j, dx_i * dy_j,
     * dy_i * dx_j, dy_i * dy_j）
     */
    __device__ __forceinline__ float4 GetProducts(int node_idx_i,
                                                  int node_idx_j,
                                                  int triangle_idx) {
        float4 sum;        // 用于存储四个导数乘积的和
        int node_offset;   // 节点在三角形中的偏移量
        float deri_i_y, deri_i_x, deri_j_y, deri_j_x;   // 存储节点的导数分量

        // 计算第一个节点相对于指定三角形的导数
        node_offset =
            FindIndex(node_rela_idx_vec_ + triangle_idx * 3, node_idx_i, 3);
        switch (node_offset) {
        case 0:
            deri_i_x = weight_;
            deri_i_y = 0;
            break;
        case 1:
            deri_i_x = weight_ * (-1 + triangle_u_);
            deri_i_y = weight_ * triangle_v_;
            break;
        case 2:
            deri_i_x = weight_ * (-triangle_u_);
            deri_i_y = weight_ * (-triangle_v_);
            break;
        }

        // 计算第二个节点相对于指定三角形的导数
        node_offset =
            FindIndex(node_rela_idx_vec_ + triangle_idx * 3, node_idx_j, 3);
        switch (node_offset) {
        case 0:
            deri_j_x = weight_;
            deri_j_y = 0;
            break;
        case 1:
            deri_j_x = weight_ * (-1 + triangle_u_);
            deri_j_y = weight_ * triangle_v_;
            break;
        case 2:
            deri_j_x = weight_ * (-triangle_u_);
            deri_j_y = weight_ * (-triangle_v_);
            break;
        }
        // 计算并累加两个节点导数的乘积
        sum = make_float4(deri_i_x * deri_j_x, deri_i_x * deri_j_y,
                          deri_i_y * deri_j_x, deri_i_y * deri_j_y);

        // 计算第一个节点相对于指定三角形的另一种导数（考虑y方向）
        node_offset =
            FindIndex(node_rela_idx_vec_ + triangle_idx * 3, node_idx_i, 3);
        switch (node_offset) {
        case 0:
            deri_i_x = 0;
            deri_i_y = weight_;
            break;
        case 1:
            deri_i_x = weight_ * (-triangle_v_);
            deri_i_y = weight_ * (-1 + triangle_u_);
            break;
        case 2:
            deri_i_x = weight_ * triangle_v_;
            deri_i_y = weight_ * (-triangle_u_);
            break;
        }

        // 计算第二个节点相对于指定三角形的另一种导数（考虑y方向）
        node_offset =
            FindIndex(node_rela_idx_vec_ + triangle_idx * 3, node_idx_j, 3);
        switch (node_offset) {
        case 0:
            deri_j_x = 0;
            deri_j_y = weight_;
            break;
        case 1:
            deri_j_x = weight_ * (-triangle_v_);
            deri_j_y = weight_ * (-1 + triangle_u_);
            break;
        case 2:
            deri_j_x = weight_ * triangle_v_;
            deri_j_y = weight_ * (-triangle_u_);
            break;
        }
        // 计算并累加两个节点在y方向导数的乘积
        sum += make_float4(deri_i_x * deri_j_x, deri_i_x * deri_j_y,
                           deri_i_y * deri_j_x, deri_i_y * deri_j_y);

        return sum;   // 返回四个导数乘积的和
    }

    /**
     * 在CUDA设备上执行的函数，用于计算并更新JTJ矩阵的值。
     * 该函数设计为内联函数，以减少函数调用开销，并在GPU线程块内执行计算和数据缩减。
     *
     * 该操作依赖于全局内存中的多个向量和数组，以及当前线程块的索引和线程索引。
     * 函数内部首先计算与当前线程块相关联的节点索引、功能数量等，
     * 然后对功能数进行循环，计算累积和，最后通过块内缩减将结果合并。
     * 根据节点是否相同，更新不同的JTJ矩阵元素。
     */

    __device__ __forceinline__ void operator()() {
        // 初始化累积和为零
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

        // 根据线程块索引计算对应的非零块索引、节点索引i和j以及功能数量
        int nz_block_idx = nz_block_idx_vec_[blockIdx.x];
        int node_idx_i   = nz_block_idx / node_num_;
        int node_idx_j   = nz_block_idx % node_num_;
        int function_num = offset_vec_[blockIdx.x + 1] -
                           offset_vec_[blockIdx.x] -
                           data_term_function_num_vec_[blockIdx.x];
        int* p_function_idx = function_idx_vec_ + offset_vec_[blockIdx.x] +
                              data_term_function_num_vec_[blockIdx.x];

        // 循环计算所有功能的累积和
        for (int i = threadIdx.x; i < function_num; i += blockDim.x) {
            float4 val = GetProducts(node_idx_i, node_idx_j,
                                     p_function_idx[i] - pixel_num_);

            sum += val;
        }

        // 使用块内缩减技术将累积和合并
        sum = BlockReduceSum(sum);

        // 只有主线程更新JTJ矩阵
        if (threadIdx.x == 0) {
            // 根据节点i和j是否相等，更新不同的JTJ矩阵元素
            if (node_idx_i == node_idx_j) {
                int start_pos =
                    pre_nz_block_num_vec_[node_idx_i * node_num_ + node_idx_j] *
                    2;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 0] + start_pos + 0] += sum.x;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 0] + start_pos + 1] += sum.y;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 1] + start_pos + 0] += sum.z;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 1] + start_pos + 1] += sum.w;
            } else {
                int start_pos =
                    pre_nz_block_num_vec_[node_idx_i * node_num_ + node_idx_j] *
                    2;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 0] + start_pos + 0] += sum.x;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 0] + start_pos + 1] += sum.y;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 1] + start_pos + 0] += sum.z;
                JTJ_a_[JTJ_ia_[node_idx_i * 2 + 1] + start_pos + 1] += sum.w;

                start_pos =
                    pre_nz_block_num_vec_[node_idx_j * node_num_ + node_idx_i] *
                    2;
                JTJ_a_[JTJ_ia_[node_idx_j * 2 + 0] + start_pos + 0] += sum.x;
                JTJ_a_[JTJ_ia_[node_idx_j * 2 + 0] + start_pos + 1] += sum.z;
                JTJ_a_[JTJ_ia_[node_idx_j * 2 + 1] + start_pos + 0] += sum.y;
                JTJ_a_[JTJ_ia_[node_idx_j * 2 + 1] + start_pos + 1] += sum.w;
            }
        }
    }
};
/**
 * 在GPU上执行SmoothTermJTJReduction操作的内核函数。
 *
 * @param stjtjr
 * 一个封装了SmoothTermJTJReduction算法的对象，其调用操作即为算法的具体实现。
 *
 * 该内核函数旨在利用CUDA并行计算的能力，高效地执行SmoothTermJTJReduction算法。
 * 通过将算法封装在对象中并传递到内核函数，可以在GPU的多个线程中同时执行算法，
 * 从而加速计算过程。
 */
__global__ void SmoothTermJTJReductionKernel(SmoothTermJTJReduction stjtjr) {
    // 在每个线程中调用SmoothTermJTJReduction对象的执行操作，
    // 实现并行计算的效果。
    stjtjr();
}

/**
 * 计算平滑项JTJ矩阵的贡献。
 *
 * @param d_JTJ_a 指向存储JTJ矩阵系数的设备内存的指针。
 * @param d_JTJ_ia 指向存储JTJ矩阵索引的设备内存的指针。
 * @param Iij 包含 NzBlock 统计数据的对象，用于JTJ计算。
 * @param d_node_vec 指向存储节点向量的设备内存的指针。
 * @param node_num 节点的数量。
 * @param pixel_num 像素的数量。
 * @param d_node_rela_idx_vec_ 指向存储节点相对索引向量的设备内存的指针。
 * @param triangle_u 三角形的u参数。
 * @param triangle_v 三角形的v参数。
 * @param weight 权重参数。
 *
 * 此函数通过在GPU上执行计算，更新JTJ矩阵的平滑项，利用了CUDA并行计算的能力。
 */
void CalcSmoothTermJTJ(float* d_JTJ_a, int* d_JTJ_ia,
                       NzBlockStatisticsForJTJ& Iij, float2* d_node_vec,
                       int node_num, int pixel_num, int* d_node_rela_idx_vec_,
                       float triangle_u, float triangle_v, float weight) {
    // 初始化SmoothTermJTJReduction结构体，用于存储计算所需的参数
    SmoothTermJTJReduction stjtjr;

    stjtjr.JTJ_a_ = d_JTJ_a;

    // 从Iij对象中拷贝需要的参数到stjtjr
    stjtjr.JTJ_ia_                     = d_JTJ_ia;
    stjtjr.nz_block_idx_vec_           = RAW_PTR(Iij.d_nz_block_idx_vec_);
    stjtjr.function_idx_vec_           = RAW_PTR(Iij.d_function_idx_vec_);
    stjtjr.offset_vec_                 = RAW_PTR(Iij.d_offset_vec_);
    stjtjr.data_term_function_num_vec_ = RAW_PTR(Iij.d_pixel_idx_num_vec_);
    stjtjr.pre_nz_block_num_vec_       = RAW_PTR(Iij.d_pre_nz_block_num_vec_);
    stjtjr.node_vec_                   = RAW_PTR(d_node_vec);
    stjtjr.node_num_                   = node_num;
    stjtjr.pixel_num_                  = pixel_num;
    stjtjr.node_rela_idx_vec_          = RAW_PTR(d_node_rela_idx_vec_);
    stjtjr.triangle_u_                 = triangle_u;
    stjtjr.triangle_v_                 = triangle_v;
    stjtjr.weight_                     = weight;

    // 设置CUDA kernel执行的网格和块大小
    int block = 128, grid = Iij.nz_block_num_;
    // 调用CUDA kernel，执行平滑项的JTJ计算
    SmoothTermJTJReductionKernel<<<grid, block>>>(stjtjr);
    // 检查CUDA执行过程中是否有错误发生
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

struct SmoothTermJTbReduction {
    float* JTb_;

    float* b_;
    int* nz_block_idx_vec_;
    int* function_idx_vec_;
    int* offset_vec_;
    int* data_term_function_num_vec_;
    int* pre_nz_block_num_vec_;
    float2* node_vec_;
    int node_num_;
    int pixel_num_;
    int* node_rela_idx_vec_;
    float triangle_u_;
    float triangle_v_;
    float weight_;

    /**
     * 获取给定节点在某个三角形中的产品和。
     *
     * @param node_idx_i 节点索引i。
     * @param triangle_idx 三角形索引。
     * @return 包含两个分量的浮点数对，表示两个方向上的导数乘积。
     */
    __device__ __forceinline__ float2 GetProducts(int node_idx_i,
                                                  int triangle_idx) {
        float2 sum;   // 用于存储两个方向上导数乘积的和

        int node_offset;   // 节点在三角形中的相对位置偏移量
        float deri_i_y, deri_i_x, deri_j_y, deri_j_x;   // 导数分量

        // 计算节点在三角形中的相对位置，并根据位置计算节点i的导数分量
        node_offset =
            FindIndex(node_rela_idx_vec_ + triangle_idx * 3, node_idx_i, 3);
        switch (node_offset) {
        case 0:
            deri_i_x = weight_;
            deri_i_y = 0;
            break;
        case 1:
            deri_i_x = weight_ * (-1 + triangle_u_);
            deri_i_y = weight_ * triangle_v_;
            break;
        case 2:
            deri_i_x = weight_ * (-triangle_u_);
            deri_i_y = weight_ * (-triangle_v_);
            break;
        }
        // 根据节点i的导数分量和预计算的系数b_计算第一个方向上的导数乘积
        sum = make_float2(deri_i_x * b_[2 * triangle_idx],
                          deri_i_y * b_[2 * triangle_idx]);

        // 重复计算节点j的导数分量和第二个方向上的导数乘积
        node_offset =
            FindIndex(node_rela_idx_vec_ + triangle_idx * 3, node_idx_i, 3);
        switch (node_offset) {
        case 0:
            deri_i_x = 0;
            deri_i_y = weight_;
            break;
        case 1:
            deri_i_x = weight_ * (-triangle_v_);
            deri_i_y = weight_ * (-1 + triangle_u_);
            break;
        case 2:
            deri_i_x = weight_ * triangle_v_;
            deri_i_y = weight_ * (-triangle_u_);
            break;
        }
        // 更新总和，包括第二个方向上的导数乘积
        sum += make_float2(deri_i_x * b_[2 * triangle_idx + 1],
                           deri_i_y * b_[2 * triangle_idx + 1]);

        return sum;   // 返回两个方向上导数乘积的和
    }

    /**
     * 在CUDA设备上执行的函数，用于计算给定块索引对应的非零元素块的贡献。
     * 该函数不接受参数，也不返回值，通过共享内存和线程块级别的同步来减少全局内存访问。
     *
     * 主要步骤包括：
     * 1. 根据块索引计算对应的非零元素块索引，并分解为节点索引i和j。
     * 2. 如果节点索引i不等于节点索引j，则函数提前返回。
     * 3. 计算作用于该块的函数数量，并准备遍历这些函数。
     * 4. 使用线程块级别的并行化，累加每个函数对该节点贡献的值。
     * 5. 使用线程块减少技术，将累加结果减少为线程块中的一个值。
     * 6. 如果当前线程是线程块中的第一个线程，则更新全局结果。
     */
    __device__ __forceinline__ void operator()() {
        // 根据块索引计算非零元素块索引，并分解为节点索引i和j
        int nz_block_idx = nz_block_idx_vec_[blockIdx.x];
        int node_idx_i   = nz_block_idx / node_num_;
        int node_idx_j   = nz_block_idx % node_num_;
        // 如果节点索引i不等于节点索引j，则函数返回
        if (node_idx_i != node_idx_j) {
            return;
        }

        // 初始化累加和
        float2 sum = {0.0f, 0.0f};

        // 计算作用于当前块的函数数量
        int function_num = offset_vec_[blockIdx.x + 1] -
                           offset_vec_[blockIdx.x] -
                           data_term_function_num_vec_[blockIdx.x];
        // 计算函数索引的起始位置
        int* p_function_idx = function_idx_vec_ + offset_vec_[blockIdx.x] +
                              data_term_function_num_vec_[blockIdx.x];

        // 并行计算所有函数的贡献
        for (int i = threadIdx.x; i < function_num; i += blockDim.x) {
            // 获取当前函数的贡献值
            float2 val =
                GetProducts(node_idx_i, p_function_idx[i] - pixel_num_);

            // 累加贡献值
            sum += val;
        }

        // 使用线程块减少技术，将累加和减少为一个值
        sum = BlockReduceSum(sum);

        // 如果当前线程是线程块中的第一个线程，则更新全局结果
        if (threadIdx.x == 0) {
            // 更新J Tb矩阵
            JTb_[node_idx_i * 2] -= sum.x;
            JTb_[node_idx_i * 2 + 1] -= sum.y;
        }
    }
};
/**
 * 在GPU上执行SmoothTermJTbReduction操作的内核函数。
 *
 * @param stjtbr
 * 一个封装了平滑项计算和减少操作的对象。该对象包含所有必要的数据和逻辑，
 *               以在设备上执行特定的平滑项计算任务。
 *
 * 该函数的设计目的是利用CUDA并行计算的能力，高效地对给定的平滑项进行计算和减少操作。
 * 通过将工作负载分布到大量的线程上，它能够显著加速处理过程，尤其适用于大规模的数据集。
 */
__global__ void SmoothTermJTbReductionKernel(SmoothTermJTbReduction stjtbr) {
    stjtbr();   // 在当前线程上执行封装在stjtbr对象中的计算和减少操作。
}

/**
 * 计算平滑项JTb
 *
 * 该函数用于在给定的节点、像素和权重等信息下，计算平滑项JTb的值。该计算涉及到非零块统计信息、节点向量等数据，
 * 并通过CUDA的并行计算能力加速计算过程。
 *
 * @param d_JTb 指向CUDA设备内存中存储JTb结果的浮点数指针。
 * @param d_b 指向CUDA设备内存中存储b的浮点数指针。
 * @param Iij NzBlockStatisticsForJTJ类型的对象，包含非零块的统计信息。
 * @param d_node_vec 指向CUDA设备内存中存储节点向量的float2类型指针。
 * @param node_num 节点的数量。
 * @param pixel_num 像素的数量。
 * @param d_node_rela_idx_vec_
 * 指向CUDA设备内存中存储节点相对索引向量的整型指针。
 * @param triangle_u 三角形的u参数。
 * @param triangle_v 三角形的v参数。
 * @param weight 权重参数。
 */
void CalcSmoothTermJTb(float* d_JTb, float* d_b, NzBlockStatisticsForJTJ& Iij,
                       float2* d_node_vec, int node_num, int pixel_num,
                       int* d_node_rela_idx_vec_, float triangle_u,
                       float triangle_v, float weight) {
    // 初始化SmoothTermJTbReduction结构体用于存储计算所需的参数
    SmoothTermJTbReduction stjtbr;

    stjtbr.JTb_ = d_JTb;

    stjtbr.b_                          = d_b;
    stjtbr.nz_block_idx_vec_           = RAW_PTR(Iij.d_nz_block_idx_vec_);
    stjtbr.function_idx_vec_           = RAW_PTR(Iij.d_function_idx_vec_);
    stjtbr.offset_vec_                 = RAW_PTR(Iij.d_offset_vec_);
    stjtbr.data_term_function_num_vec_ = RAW_PTR(Iij.d_pixel_idx_num_vec_);
    stjtbr.pre_nz_block_num_vec_       = RAW_PTR(Iij.d_pre_nz_block_num_vec_);
    stjtbr.node_vec_                   = RAW_PTR(d_node_vec);
    stjtbr.node_num_                   = node_num;
    stjtbr.pixel_num_                  = pixel_num;
    stjtbr.node_rela_idx_vec_          = RAW_PTR(d_node_rela_idx_vec_);
    stjtbr.triangle_u_                 = triangle_u;
    stjtbr.triangle_v_                 = triangle_v;
    stjtbr.weight_                     = weight;

    // 设置CUDA kernel的执行配置，并启动kernel进行计算
    int block = 128, grid = Iij.nz_block_num_;
    SmoothTermJTbReductionKernel<<<grid, block>>>(stjtbr);

    // 检查CUDA执行过程中是否有错误发生
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

struct ZeroTermJTJReduction {
    float* JTJ_a_;

    int* JTJ_ia_;
    int* pre_nz_block_num_vec_;
    float2* node_vec_;
    float2* original_node_vec_;
    int node_num_;
    float weight_;

    /**
     * 在CUDA设备上执行的函数，用于更新JTJ矩阵的特定元素。
     * 该函数设计为内联，以减少函数调用开销，并在GPU线程块的每个线程上运行。
     * 每个线程负责更新JTJ矩阵中的一部分元素，基于给定的权重。
     *
     * 无参数
     * 无显式返回值，但会对全局数组JTJ_a_和JTJ_ia_进行修改。
     */
    __device__ __forceinline__ void operator()() {
        // 计算当前线程处理的节点索引
        int node_idx = threadIdx.x + blockIdx.x * blockDim.x;

        // 如果节点索引超出节点总数，则提前返回
        if (node_idx >= node_num_)
            return;

        // 计算权重的平方，用于后续更新操作
        float weight_square = weight_ * weight_;

        // 计算在JTJ矩阵中的起始位置
        int start_pos =
            pre_nz_block_num_vec_[node_idx * node_num_ + node_idx] * 2;

        // 根据节点索引和起始位置，更新JTJ矩阵的两个元素
        JTJ_a_[JTJ_ia_[node_idx * 2 + 0] + start_pos + 0] += weight_square;
        JTJ_a_[JTJ_ia_[node_idx * 2 + 1] + start_pos + 1] += weight_square;
    }
};
/**
 * 在GPU上执行零项JTJ缩减操作的内核函数。
 *
 * @param ztjtjr
 * 一个封装了零项JTJ缩减具体实现的对象。该对象包含所有必要的数据和操作，
 *               通过此参数传递到内核函数中，并在内核函数中被调用执行具体的缩减操作。
 *
 * 该内核函数的设计目的是利用GPU的并行计算能力，高效地对大型矩阵的零项JTJ部分进行缩减处理。
 * 具体的处理逻辑和算法被封装在传入的`ZeroTermJTJReduction`对象中，内核函数通过调用该对象的
 * 操作符()来执行具体的计算任务。
 */
__global__ void ZeroTermJTJReductionKernel(ZeroTermJTJReduction ztjtjr) {
    ztjtjr();   // 调用传入对象的执行操作，完成零项JTJ的缩减计算。
}

/**
 * 计算零项JTJ矩阵的缩减操作。
 *
 * @param d_JTJ_a 指向CUDA设备上存储JTJ矩阵系数的浮点数数组。
 * @param d_JTJ_ia 指向CUDA设备上存储JTJ矩阵索引的整数数组。
 * @param Iij 包含JTJ计算所需预处理非零块统计信息的对象。
 * @param d_original_node_vec 指向CUDA设备上存储原始节点向量的浮点数数组。
 * @param d_node_vec 指向CUDA设备上存储当前节点向量的浮点数数组。
 * @param node_num 节点的数量。
 * @param weight 权重参数。
 *
 * 此函数不返回值，但会在CUDA设备上更新JTJ矩阵的缩减表示。
 */
void CalcZeroTermJTJ(float* d_JTJ_a, int* d_JTJ_ia,
                     NzBlockStatisticsForJTJ& Iij, float2* d_original_node_vec,
                     float2* d_node_vec, int node_num, float weight) {
    // 初始化用于JTJ矩阵缩减的结构体
    ZeroTermJTJReduction ztjtjr;

    // 设置结构体中的成员变量
    ztjtjr.JTJ_a_ = d_JTJ_a;

    ztjtjr.JTJ_ia_               = d_JTJ_ia;
    ztjtjr.pre_nz_block_num_vec_ = RAW_PTR(Iij.d_pre_nz_block_num_vec_);
    ztjtjr.original_node_vec_    = RAW_PTR(d_original_node_vec);
    ztjtjr.node_vec_             = RAW_PTR(d_node_vec);
    ztjtjr.node_num_             = node_num;
    ztjtjr.weight_               = weight;

    // 设置CUDA kernel执行的网格和块大小
    int block = 256, grid = (node_num + block - 1) / block;
    // 启动CUDA kernel来执行零项JTJ矩阵的缩减计算
    ZeroTermJTJReductionKernel<<<grid, block>>>(ztjtjr);
    // 检查并处理CUDA执行过程中可能出现的错误
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
