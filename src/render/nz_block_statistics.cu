#include "render/nz_block_statistics.cuh"
#include "util/cuda_utils.h"
#include "util/innoreal_timer.hpp"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

/**
 * FillPixelIdxWrapper 结构体用于封装在 CUDA 设备上执行的，
 * 用于填充像素索引和函数索引的操作。
 *
 * @param nz_block_idx 指向设备内存的指针，用于存储非零块索引。
 * @param function_idx 指向设备内存的指针，用于存储对应像素的函数索引。
 * @param pixel_num 像素的数量。
 * @param node_num 节点的数量。
 * @param nz_block_num_each_pixel 每个像素对应的非零块数量。
 * @param pixel_rela_idx_vec
 * 指向设备内存的指针，存储每个像素相对应的节点索引向量。
 */
struct FillPixelIdxWrapper {
    int* nz_block_idx;
    int* function_idx;

    int pixel_num;
    int node_num;
    int nz_block_num_each_pixel;
    int* pixel_rela_idx_vec;

    /**
     * 运算符()重载为 CUDA 设备函数，用于并行填充 nz_block_idx 和 function_idx
     * 数组。
     *
     * @note 该函数设计为被 CUDA kernel 调用，以并行方式处理所有像素。
     */
    __device__ void operator()() {
        int pixel_idx = threadIdx.x + blockDim.x * blockIdx.x;

        // 只处理属于当前块的像素
        if (pixel_idx < pixel_num) {
            int node_i, node_j, block_idx;
            int counter      = 0;
            int* pixel_relas = pixel_rela_idx_vec + pixel_idx * 4;

            // 填充非零块索引
            for (int neib_idx_i = 0; neib_idx_i < 4; ++neib_idx_i) {
                node_i = pixel_relas[neib_idx_i];
                for (int neib_idx_j = neib_idx_i; neib_idx_j < 4;
                     ++neib_idx_j) {
                    node_j = pixel_relas[neib_idx_j];
                    // 根据节点索引计算非零块索引，并填充
                    if (node_i < node_j) {
                        block_idx = node_j * node_num + node_i;
                    } else {
                        block_idx = node_i * node_num + node_j;
                    }

                    nz_block_idx[pixel_idx * nz_block_num_each_pixel +
                                 counter] = block_idx;
                    ++counter;
                }
            }

            // 填充函数索引，所有非零块对应同一个像素时函数索引都为该像素的索引
            for (int iter = 0; iter < nz_block_num_each_pixel; ++iter) {
                function_idx[pixel_idx * nz_block_num_each_pixel + iter] =
                    pixel_idx;
            }
        }
    }
};

/**
 * 在CUDA设备上执行的填充像素索引的内核函数。
 *
 * @param op
 * 一个封装了实际操作的FillPixelIdxWrapper对象。该对象的调用（op()）将在每个线程上执行，
 *           允许对像素索引进行操作和填充。
 *
 * 该函数设计为在GPU上运行，通过调用传入的操作对象，对图像的像素索引进行批量处理。
 * 每个线程将调用一次操作对象，以处理相应的像素索引。
 */
__global__ void FillPixelIdxKernel(FillPixelIdxWrapper op) {
    // 在这里，我们对操作对象进行调用，以便在当前线程上执行实际的像素索引填充逻辑。
    op();
}

/**
 * 结构体FillTriangleIdxWrapper用于封装填充三角形索引的操作。
 * 它通过在CUDA设备上运行的operator()成员函数来实现具体的计算。
 */
struct FillTriangleIdxWrapper {
    int* nz_block_idx;   // 非零块索引数组
    int* function_idx;   // 函数索引数组

    int pixel_num;                    // 像素数量
    int triangle_num;                 // 三角形数量
    int node_num;                     // 节点数量
    int nz_block_num_each_triangle;   // 每个三角形的非零块数量
    int* node_rela_idx_vec;           // 节点关系索引向量

    /**
     * __device__限定符表示此函数可以在CUDA设备上运行。
     * 函数负责根据三角形索引填充非零块索引和函数索引数组。
     */
    __device__ void operator()() {
        int triangle_idx =
            threadIdx.x + blockDim.x * blockIdx.x;   // 计算当前处理的三角形索引

        if (triangle_idx < triangle_num) {   // 检查是否在三角形数量范围内
            int node_i, node_j, block_idx;
            int counter = 0;
            int* node_relas =
                node_rela_idx_vec +
                triangle_idx * 3;   // 计算当前三角形的节点关系索引起始位置
            // 遍历节点对，生成非零块索引
            for (int neib_idx_i = 0; neib_idx_i < 3; ++neib_idx_i) {
                node_i = node_relas[neib_idx_i];
                for (int neib_idx_j = neib_idx_i; neib_idx_j < 3;
                     ++neib_idx_j) {
                    node_j = node_relas[neib_idx_j];
                    // 计算块索引，并根据节点关系填充非零块索引数组
                    if (node_i < node_j) {
                        block_idx = node_j * node_num + node_i;
                    } else {
                        block_idx = node_i * node_num + node_j;
                    }

                    nz_block_idx[triangle_idx * nz_block_num_each_triangle +
                                 counter] = block_idx;
                    ++counter;
                }
            }
            // 填充函数索引数组
            for (int iter = 0; iter < nz_block_num_each_triangle; ++iter) {
                function_idx[triangle_idx * nz_block_num_each_triangle + iter] =
                    triangle_idx + pixel_num;
            }
        }
    }
};

/**
 * __global__限定符表示此函数可以在CUDA设备上作为一个全局函数调用。
 * 它的作用是启动FillTriangleIdxWrapper操作的并行执行。
 */
__global__ void FillTriangleIdxKernel(FillTriangleIdxWrapper op) {
    op();   // 调用FillTriangleIdxWrapper的operator()成员函数
}

/**
 * 结构体CalcOffsetFlagWrapper用于封装计算偏移标志的操作。
 * 它包含非零块索引数组和偏移标志数组，以及操作这些数据的设备内函数。
 */
struct CalcOffsetFlagWrapper {
    int* offset_flag;           // 偏移标志数组的指针
    int nz_block_idx_vec_len;   // 非零块索引向量的长度
    int* nz_block_idx;          // 非零块索引数组的指针

    /**
     * 设备内函数，用于计算偏移标志。
     * 根据非零块索引的值，确定是否需要设置对应的偏移标志。
     */
    __device__ void operator()() {
        int idx = threadIdx.x + blockDim.x * blockIdx.x +
                  1;   // 计算当前线程处理的索引

        // 当索引在有效范围内且当前块索引与前一个块索引不同时，设置偏移标志
        if (idx > 0 && idx < nz_block_idx_vec_len) {
            if (nz_block_idx[idx] != nz_block_idx[idx - 1]) {
                offset_flag[idx] = idx;
            }
        }
    }
};

/**
 * 核心函数，用于在设备上启动计算偏移标志的操作。
 * @param op 包含偏移标志计算操作的CalcOffsetFlagWrapper对象。
 */
__global__ void CalcOffsetFlagKernel(CalcOffsetFlagWrapper op) {
    op();   // 执行计算偏移标志的操作
}

/**
 * 结构体 CalcNzBlockIdxWrapper 用于封装计算非零块索引的操作。
 * 它包含用于标识非零块索引的标志数组、非零块索引向量的长度和非零块索引数组。
 */
struct CalcNzBlockIdxWrapper {
    int*
        nz_block_idx_flag;   // 标志数组，用于标识索引位置是否为非零块的起始位置。
    int nz_block_idx_vec_len;   // 非零块索引数组的长度。
    int* nz_block_idx;          // 非零块索引数组。

    /**
     * 在设备上运行的运算符函数，用于计算非零块索引的标志。
     * 它根据线程的索引位置检查并更新非零块索引标志数组。
     */
    __device__ void operator()() {
        int idx = threadIdx.x + blockDim.x * blockIdx.x +
                  1;   // 计算当前线程对应的索引位置。
        if (idx > 0 &&
            idx < nz_block_idx_vec_len) {   // 确保索引位置在有效范围内。
            if (nz_block_idx[idx] !=
                nz_block_idx
                    [idx -
                     1]) {   // 如果当前索引位置的非零块索引与前一个不同，则设置标志。
                nz_block_idx_flag[idx] = nz_block_idx[idx];
            }
        }
    }
};

/**
 * __global__ 函数 CalcNzBlockIdxKernel 用于启动并行计算以确定非零块索引的标志。
 * 它通过传递 CalcNzBlockIdxWrapper 实例来执行操作。
 *
 * @param op CalcNzBlockIdxWrapper 实例，包含执行操作所需的所有数据和函数。
 */
__global__ void CalcNzBlockIdxKernel(CalcNzBlockIdxWrapper op) {
    op();   // 调用运算符函数以执行计算。
}

/**
 * 结构体 SetNzFlagWrapper 用于封装非零标志设置的相关数据和操作。
 *
 * 成员变量:
 * - nz_flag: 指向布尔型数组的指针，用于标记是否为非零。
 * - nz_idx_num: 非零索引的数量。
 * - nz_block_idx: 存储非零索引的数组。
 * - node_num: 节点数。
 */
struct SetNzFlagWrapper {
    bool* nz_flag;

    int nz_idx_num;
    int* nz_block_idx;
    int node_num;

    /**
     * __device__ 函数，用于在设备上运行，设置指定索引的非零标志。
     * 这个函数通过线程块和线程索引来遍历 nz_block_idx 数组，并设置对应的
     * nz_flag。
     */
    __device__ void operator()() {
        int idx =
            threadIdx.x + blockDim.x * blockIdx.x;   // 计算当前线程处理的索引
        if (idx < nz_idx_num) {   // 检查索引是否在有效范围内
            int index = nz_block_idx[idx];   // 获取当前线程处理的非零索引
            int seri_i     = index / node_num;            // 计算序列号 i
            int seri_j     = index - seri_i * node_num;   // 计算序列号 j
            nz_flag[index] = true;   // 设置当前索引的非零标志
            nz_flag[seri_j * node_num + seri_i] =
                true;   // 设置对应的另一个索引的非零标志
        }
    }
};

/**
 * __global__ 函数，用于启动 SetNzFlagWrapper 中定义的操作。
 * 它将操作分发到多个线程上，以并行方式设置非零标志。
 *
 * 参数:
 * - snf_wrapper: SetNzFlagWrapper 实例，包含要操作的数据和设置。
 */
__global__ void SetNzFlagKernel(SetNzFlagWrapper snf_wrapper) {
    snf_wrapper();   // 调用 SetNzFlagWrapper 的 operator()，执行非零标志设置
}

/**
 * 结构体 GetNnzPreWrapper 用于计算非零元素前置和的预处理。
 * 其中包含用于计算的多个指针和节点数量。
 */
struct GetNnzPreWrapper {
    int* pre_nz_block_num;   // 存储每个块的非零元素前置和
    int* row_offset;         // 存储每一行的偏移量，用于快速访问
    bool* nz_flag;           // 标记每个节点是否包含非零元素
    int node_num;            // 节点数量

    /**
     * 运算符()重载为设备内函数，用于执行非零元素前置和的计算。
     * 利用块内每个warp的合作完成prefix_sum。
     */
    __device__ void operator()() {
        __shared__ int s_pre[1];   // 共享内存用于存储局部前缀和
        if (threadIdx.x == 0) {   // 线程0初始化共享内存和块级非零元素前置和
            s_pre[0]                                = 0;
            pre_nz_block_num[node_num * blockIdx.x] = 0;
        }
        // __syncwarp();  // 同步warp

        bool* flag_ptr = nz_flag + node_num * blockIdx.x;   // 计算标志指针
        // nnz_pre_ptr需要的是exclude scan, 加一个偏移在include scan上实现
        int* nnz_pre_ptr = pre_nz_block_num + node_num * blockIdx.x +
                           1;   // 计算非零元素前置和指针
        int value, lane_id;

        // 循环内为warp内include scan（前缀和），blockDim.x=32
        for (int iter = threadIdx.x; iter < node_num - 1; iter += blockDim.x) {
            lane_id = threadIdx.x & 0x1f;   // 计算线程在warp内的ID
            value   = flag_ptr[iter];       // 获取当前节点的标志
            for (int i = 1; i <= 32;
                 i *= 2) {   // 使用shfl_up_sync进行warp内的前缀和计算
                int n = __shfl_up_sync(__activemask(), value, i, 32);
                if ((lane_id & 31) >= i) {
                    value += n;
                }
            }
            nnz_pre_ptr[iter] = s_pre[0] + value;   // 更新非零元素前置和
            if (threadIdx.x == blockDim.x - 1) {
                s_pre[0] += value;   // 如果是最后一个线程，则更新全局前缀和
            }
            __syncwarp();   // 同步warp
        }

        if (threadIdx.x == 0) {   // 线程0负责更新行偏移量
            if (flag_ptr[node_num - 1]) {
                row_offset[blockIdx.x] = nnz_pre_ptr[node_num - 2] + 1;
            } else {
                row_offset[blockIdx.x] = nnz_pre_ptr[node_num - 2];
            }
        }
    }
};

/**
 * __global__函数 GetNnzPreKernel 用于启动GetNnzPreWrapper结构体中的计算。
 * @param gnp_wrapper GetNnzPreWrapper结构体的实例，包含所有需要的参数和数据。
 */
__global__ void GetNnzPreKernel(GetNnzPreWrapper gnp_wrapper) {
    gnp_wrapper();   // 调用GetNnzPreWrapper的运算符()执行计算
}

/**
 * 结构体 CalcPixelIdxNumForEachNzBlockWrapper
 * 用于计算每个非零索引块中的像素索引数量。
 *
 * @param pixel_idx_num_vec
 * 指向整型数组的指针，用于存储每个非零索引块中的像素索引数量。
 * @param nz_idx_num 非零索引的数量。
 * @param pixel_num 像素的总数。
 * @param function_idx_vec 存储功能索引的整型数组。
 * @param offset_vec 存储偏移量的整型数组，用于定位功能索引在 function_idx_vec
 * 中的起始位置。
 */
struct CalcPixelIdxNumForEachNzBlockWrapper {
    int* pixel_idx_num_vec;

    int nz_idx_num;
    int pixel_num;
    int* function_idx_vec;
    int* offset_vec;

    /**
     * 在设备上运行的运算符函数，用于计算每个非零索引块中的像素索引数量。
     */
    __device__ void operator()() {
        __shared__ int* s_func_list_ptr;   // 共享内存中指向功能列表指针的变量
        __shared__ int s_num_fun;   // 共享内存中存储功能数量的变量
        __shared__ int
            s_res_each_warp[4];   // 共享内存数组，用于存储每个 warp 的结果

        // 当当前块索引小于非零索引数量时执行计算
        if (blockIdx.x < nz_idx_num) {
            if (threadIdx.x == 0) {   // 确定共享内存变量的初始值
                s_func_list_ptr = function_idx_vec + offset_vec[blockIdx.x];
                s_num_fun = offset_vec[blockIdx.x + 1] - offset_vec[blockIdx.x];
            }
            if (threadIdx.x <
                4) {   // 初始化_each_warp[threadIdx.x] = warp 结果数组
                s_res_each_warp[threadIdx.x] = 0;   // 确保所有 0;
            }
            __syncthreads();

            bool is_pixel_idx;
            unsigned vote_result;
            int warp_id = threadIdx.x / 32;   // 计算当前线程所在的 warp ID
            // 使用投票机制统计每个 warp 中具有像素索引的功能数量
            for (int iter = threadIdx.x; iter < s_num_fun; iter += blockDim.x) {
                is_pixel_idx = (s_func_list_ptr[iter] < pixel_num);
                vote_result  = __ballot_sync(__activemask(), is_pixel_idx);
                // 计算具有像素索引的功能数量
                if (threadIdx.x == warp_id * 32 && vote_result > 0) {
                    s_res_each_warp[warp_id] +=
                        int(log2f(float(vote_result) + 1.f));
                }
                __syncthreads();   // 确保所有线程完成当前迭代
            }
            if (threadIdx.x == 0) {   // 将结果汇总并写回到全局内存
                // 这里vote_result的1一定是连续，且1在前0在后，小端模式存储，比如在内存中为11110000...
                pixel_idx_num_vec[blockIdx.x] =
                    s_res_each_warp[0] + s_res_each_warp[1] +
                    s_res_each_warp[2] + s_res_each_warp[3];
            }
        }
    }
};

/**
 * 核函数 CalcPixelIdxNumForEachNzBlockKernel 用于启动
 * CalcPixelIdxNumForEachNzBlockWrapper 运算。
 *
 * @param op CalcPixelIdxNumForEachNzBlockWrapper 实例。
 */
__global__ void
CalcPixelIdxNumForEachNzBlockKernel(CalcPixelIdxNumForEachNzBlockWrapper op) {
    op();   // 执行计算
}

/**
    @brief:
   此函数用于计算非零块（nz_block）及其相关函数，首先根据像素和三角形数量计算非零块索引和函数索引，并将其存储在相应的设备向量中。
    然后，将非零块索引向量转换为CSR稀疏矩阵格式的行偏移向量和非零元素数量向量。

    @param:
    - pixel_num: 像素数量
    - triangle_num: 三角形数量
    - node_num: 节点数量
    - d_pixel_rela_idx_vec: 像素相关索引的设备指针
    - d_pixel_rela_weight_vec: 像素相关权重的设备指针
    - d_node_rela_idx_vec: 节点相关索引的设备指针
    - d_node_rela_weight_vec: 节点相关权重的设备指针
    */
void NzBlockStatisticsForJTJ::CalcNzBlocksAndFunctions(
    int pixel_num, int triangle_num, int node_num, int* d_pixel_rela_idx_vec,
    float* d_pixel_rela_weight_vec, int* d_node_rela_idx_vec,
    float* d_node_rela_weight_vec) {
    /**
    第一步：
      获取4个vector，分别为：d_nz_block_idx_vec，d_func_idx_vec，d_pixel_idx_num_vec_，d_offset_vec_

      1. d_nz_block_idx_vec是一个列表， 里面每个元素是nz_block的index，
    nz_block_idx的范围是[0， node_num*node_num-1]

      2. d_func_idx_vec是一个列表， 里面每个元素是func_idx，
    这里func_idx具体为pixel_idx和triangle_idx， pixel_idx排在triangle_idx前面，
        每一个nz_block的pixel_idx数量是不固定的，
    其数量存储在计算d_pixel_idx_num_vec_中

      3.
    d_nz_block_idx_vec中的每个nz_block和d_func_idx_vec中的一系列func_idx对应，其对应关系存储在d_offset_vec_中，具体的，如果d_offset_vec_
        为[0， 5， 20， 30]，则d_func_idx_vec中的[0，
    5)为d_nz_block_idx_vec[0]对应的func_idx，以此类推

    第二步：
      将d_nz_block_idx_vec所表达的稀疏矩阵转换为d_row_offset_vec_和d_pre_nz_block_num_vec_，其中，d_row_offset_vec_为CSR稀疏矩阵的
        行索引，d_pre_nz_block_num_vec_是一个node_num *
    node_num的向量，用于将JTJ矩阵中对应的nz_block_idx转换到CSR稀疏矩阵的当前行前面非零
        元素的个数，用于写如数据到CSR矩阵。此时，d_row_offset_vec_[i]即为JTJ矩阵第i行的CSR行索引，d_pre_nz_block_num_vec_[nz_block_idx]即为nz_block_idx对应的列索引
      注意：d_nz_block_idx_vec中只存储了对称矩阵的下三角非零块（包含对角线）
      */

    // 第一步：获取4个vector
    int nz_block_num_each_pixel =
        (4 * 4 + 4) /
        2; /** 每个pixel包含4个node，形成个(4*4+4)/2下三角中的非零块 */
    int nz_block_num_each_triangle =
        (3 * 3 + 3) /
        2; /** 每一个triangle包含3个node, 形成个(3*3+3)/2下三角中的非零块 */
    int nz_block_idx_vec_len = pixel_num * nz_block_num_each_pixel +
                               triangle_num * nz_block_num_each_triangle;

    thrust::device_vector<int> d_nz_block_idx_vec_duplicate(
        nz_block_idx_vec_len);
    d_function_idx_vec_.resize(nz_block_idx_vec_len);

    // 将pixel_idx以及对应的nz_block填充到d_nz_block_idx_vec_duplicate中
    // 使用像素索引填充非零块索引和函数索引
    int block = 256, grid = (pixel_num + block - 1) / block;
    FillPixelIdxWrapper fill_pixel_idx_wrapper;
    fill_pixel_idx_wrapper.nz_block_idx = RAW_PTR(d_nz_block_idx_vec_duplicate);
    fill_pixel_idx_wrapper.function_idx = RAW_PTR(d_function_idx_vec_);
    fill_pixel_idx_wrapper.pixel_num    = pixel_num;
    fill_pixel_idx_wrapper.node_num     = node_num;
    fill_pixel_idx_wrapper.nz_block_num_each_pixel = nz_block_num_each_pixel;
    fill_pixel_idx_wrapper.pixel_rela_idx_vec = RAW_PTR(d_pixel_rela_idx_vec);
    if (grid > 0) {
        FillPixelIdxKernel<<<grid, block>>>(fill_pixel_idx_wrapper);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 将triangle_idx以及对应的nz_block填充到d_nz_block_idx_vec_duplicate中
    // 使用三角形索引填充非零块索引和函数索引
    block = 64, grid = (triangle_num + block - 1) / block;
    FillTriangleIdxWrapper fill_triangle_idx_wrapper;
    fill_triangle_idx_wrapper.nz_block_idx =
        RAW_PTR(d_nz_block_idx_vec_duplicate) +
        pixel_num * nz_block_num_each_pixel;
    fill_triangle_idx_wrapper.function_idx =
        RAW_PTR(d_function_idx_vec_) + pixel_num * nz_block_num_each_pixel;
    fill_triangle_idx_wrapper.pixel_num    = pixel_num;
    fill_triangle_idx_wrapper.triangle_num = triangle_num;
    fill_triangle_idx_wrapper.node_num     = node_num;
    fill_triangle_idx_wrapper.nz_block_num_each_triangle =
        nz_block_num_each_triangle;
    fill_triangle_idx_wrapper.node_rela_idx_vec = RAW_PTR(d_node_rela_idx_vec);
    if (grid > 0) {
        FillTriangleIdxKernel<<<grid, block>>>(fill_triangle_idx_wrapper);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 对填充后的非零块索引和函数索引进行排序，以便后续处理
    thrust::stable_sort_by_key(d_nz_block_idx_vec_duplicate.begin(),
                               d_nz_block_idx_vec_duplicate.end(),
                               d_function_idx_vec_.begin());

    /** 计算d_offset_vec_，nz_idx_num_ */
    // 计算每个非零块相对应的函数索引偏移量
    thrust::device_vector<int> d_temp_array(nz_block_idx_vec_len);
    checkCudaErrors(cudaMemset(RAW_PTR(d_temp_array), 0,
                               nz_block_idx_vec_len * sizeof(int)));
    block = 256, grid = (nz_block_idx_vec_len + block - 1) / block;
    CalcOffsetFlagWrapper calc_offset_wrapper;
    calc_offset_wrapper.offset_flag          = RAW_PTR(d_temp_array);
    calc_offset_wrapper.nz_block_idx_vec_len = nz_block_idx_vec_len;
    calc_offset_wrapper.nz_block_idx = RAW_PTR(d_nz_block_idx_vec_duplicate);
    CalcOffsetFlagKernel<<<grid, block>>>(calc_offset_wrapper);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    /** 去除0，保留第一个0 */
    // 从偏移量数组中提取非零块数量并计算最终的偏移量向量
    auto new_end =
        thrust::remove(d_temp_array.begin() + 1, d_temp_array.end(), int(0));
    nz_block_num_ = new_end - d_temp_array.begin();
    d_offset_vec_.resize(nz_block_num_ + 1);
    d_temp_array[nz_block_num_] =
        nz_block_idx_vec_len; /** 最后一位填充为总长度 */
    checkCudaErrors(cudaMemcpy(RAW_PTR(d_offset_vec_), RAW_PTR(d_temp_array),
                               (nz_block_num_ + 1) * sizeof(int),
                               cudaMemcpyDeviceToDevice));

    /** 计算d_nz_block_idx_vec_ */
    // 根据排序后的非零块索引和偏移量计算最终的非零块索引向量
    checkCudaErrors(cudaMemset(RAW_PTR(d_temp_array), 0,
                               nz_block_idx_vec_len * sizeof(int)));
    block = 256, grid = (nz_block_idx_vec_len + block - 1) / block;
    CalcNzBlockIdxWrapper calc_nz_block_idx_wrapper;
    calc_nz_block_idx_wrapper.nz_block_idx_flag    = RAW_PTR(d_temp_array);
    calc_nz_block_idx_wrapper.nz_block_idx_vec_len = nz_block_idx_vec_len;
    calc_nz_block_idx_wrapper.nz_block_idx =
        RAW_PTR(d_nz_block_idx_vec_duplicate);
    CalcNzBlockIdxKernel<<<grid, block>>>(calc_nz_block_idx_wrapper);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    /** 去除0，保留第一个元素，如果nz_block_idx中包含0，则第一个元素一定是0 */
    // 从计算得到的非零块索引中去除重复项
    thrust::remove(d_temp_array.begin() + 1, d_temp_array.end(), 0);
    d_nz_block_idx_vec_.resize(nz_block_num_);
    checkCudaErrors(
        cudaMemcpy(RAW_PTR(d_nz_block_idx_vec_), RAW_PTR(d_temp_array),
                   nz_block_num_ * sizeof(int), cudaMemcpyDeviceToDevice));

    /** 计算d_pixel_idx_num_vec_ */
    // 计算每个非零块对应的像素索引数量
    d_pixel_idx_num_vec_.resize(nz_block_num_, 0);

    CalcPixelIdxNumForEachNzBlockWrapper calc_pixel_idx_num_wrapper;
    calc_pixel_idx_num_wrapper.pixel_idx_num_vec =
        RAW_PTR(d_pixel_idx_num_vec_);
    calc_pixel_idx_num_wrapper.nz_idx_num       = nz_block_num_;
    calc_pixel_idx_num_wrapper.pixel_num        = pixel_num;
    calc_pixel_idx_num_wrapper.function_idx_vec = RAW_PTR(d_function_idx_vec_);
    calc_pixel_idx_num_wrapper.offset_vec       = RAW_PTR(d_offset_vec_);
    block = 32 * 4, grid = nz_block_num_;   // 每个block包含128个线程，4个warp
    /**
     * 计算每个非零块对应的像素索引数量
     * 使用CUDA
     * Kernel调用calc_pixel_idx_num_wrapper函数，计算非零块中的像素索引数量。
     */
    CalcPixelIdxNumForEachNzBlockKernel<<<grid, block>>>(
        calc_pixel_idx_num_wrapper);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    /**
     * 将稀疏矩阵的非零块信息转换为CSR格式
     * 首先创建一个布尔类型的设备向量d_nz_flag，用于标记矩阵中非零元素的位置。
     * 然后通过SetNzFlagKernel
     * kernel调用set_nz_flag_wrapper函数，将非零块的位置标记在d_nz_flag中。
     */
    thrust::device_vector<bool> d_nz_flag(node_num * node_num, false);
    // 创建一个NxN的标志位矩阵，非零块的位置标记为1
    SetNzFlagWrapper set_nz_flag_wrapper;
    set_nz_flag_wrapper.nz_flag      = RAW_PTR(d_nz_flag);
    set_nz_flag_wrapper.nz_idx_num   = nz_block_num_;
    set_nz_flag_wrapper.node_num     = node_num;
    set_nz_flag_wrapper.nz_block_idx = RAW_PTR(d_nz_block_idx_vec_);
    block = 256, grid = (nz_block_num_ + block - 1) / block;
    SetNzFlagKernel<<<grid, block>>>(set_nz_flag_wrapper);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    /**
     * 计算d_pre_nz_idx_num_vec_和d_row_offset_vec_
     * 这一步用于预先计算每个非零块之前有多少非零块，以及为CSR格式的稀疏矩阵计算row_offset。
     */
    thrust::device_vector<int> d_row_offset_vec(node_num);
    d_row_offset_vec_.resize(node_num + 1, 0);
    d_pre_nz_block_num_vec_.resize(node_num * node_num);

    GetNnzPreWrapper get_nnz_pre_wrapper;
    get_nnz_pre_wrapper.pre_nz_block_num = RAW_PTR(d_pre_nz_block_num_vec_);
    get_nnz_pre_wrapper.row_offset       = RAW_PTR(d_row_offset_vec);
    get_nnz_pre_wrapper.nz_flag          = RAW_PTR(d_nz_flag);
    get_nnz_pre_wrapper.node_num         = node_num;
    block = 32, grid = node_num;   // 每个block包含32个线程，1个warp
    GetNnzPreKernel<<<grid, block>>>(get_nnz_pre_wrapper);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    /**
     * 计算row_offset_vec_的累积和
     * 使用thrust库的inclusive_scan函数，计算d_row_offset_vec的累积和，得到row_offset_vec_。
     * 这是CSR格式稀疏矩阵构造的最后一步。
     */
    thrust::inclusive_scan(d_row_offset_vec.begin(), d_row_offset_vec.end(),
                           d_row_offset_vec_.begin() + 1);
}
