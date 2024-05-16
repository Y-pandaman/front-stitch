#include "cuda_utils.h"
#include "innoreal_timer.hpp"
#include "nz_block_statistics.cuh"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

struct FillPixelIdxWrapper {
    int* nz_block_idx;
    int* function_idx;

    int pixel_num;
    int node_num;
    int nz_block_num_each_pixel;
    int* pixel_rela_idx_vec;

    __device__ void operator()() {
        int pixel_idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (pixel_idx < pixel_num) {
            int node_i, node_j, block_idx;
            int counter      = 0;
            int* pixel_relas = pixel_rela_idx_vec + pixel_idx * 4;
            // fill nz_idx
            for (int neib_idx_i = 0; neib_idx_i < 4; ++neib_idx_i) {
                node_i = pixel_relas[neib_idx_i];
                for (int neib_idx_j = neib_idx_i; neib_idx_j < 4;
                     ++neib_idx_j) {
                    node_j = pixel_relas[neib_idx_j];
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
            // fill function_idx
            for (int iter = 0; iter < nz_block_num_each_pixel; ++iter) {
                function_idx[pixel_idx * nz_block_num_each_pixel + iter] =
                    pixel_idx;
            }
        }
    }
};
__global__ void FillPixelIdxKernel(FillPixelIdxWrapper op) {
    op();
}

struct FillTriangleIdxWrapper {
    int* nz_block_idx;
    int* function_idx;

    int pixel_num;
    int triangle_num;
    int node_num;
    int nz_block_num_each_triangle;
    int* node_rela_idx_vec;

    __device__ void operator()() {
        int triangle_idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (triangle_idx < triangle_num) {
            int node_i, node_j, block_idx;
            int counter     = 0;
            int* node_relas = node_rela_idx_vec + triangle_idx * 3;
            for (int neib_idx_i = 0; neib_idx_i < 3; ++neib_idx_i) {
                node_i = node_relas[neib_idx_i];
                for (int neib_idx_j = neib_idx_i; neib_idx_j < 3;
                     ++neib_idx_j) {
                    node_j = node_relas[neib_idx_j];
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
            // fill function_idx
            for (int iter = 0; iter < nz_block_num_each_triangle; ++iter) {
                function_idx[triangle_idx * nz_block_num_each_triangle + iter] =
                    triangle_idx + pixel_num;
            }
        }
    }
};
__global__ void FillTriangleIdxKernel(FillTriangleIdxWrapper op) {
    op();
}

struct CalcOffsetFlagWrapper {
    int* offset_flag;

    int nz_block_idx_vec_len;
    int* nz_block_idx;

    __device__ void operator()() {
        int idx = threadIdx.x + blockDim.x * blockIdx.x + 1;

        if (idx > 0 && idx < nz_block_idx_vec_len) {
            if (nz_block_idx[idx] != nz_block_idx[idx - 1]) {
                offset_flag[idx] = idx;
            }
        }
    }
};
__global__ void CalcOffsetFlagKernel(CalcOffsetFlagWrapper op) {
    op();
}

struct CalcNzBlockIdxWrapper {
    int* nz_block_idx_flag;

    int nz_block_idx_vec_len;
    int* nz_block_idx;

    __device__ void operator()() {
        int idx = threadIdx.x + blockDim.x * blockIdx.x + 1;
        if (idx > 0 && idx < nz_block_idx_vec_len) {
            if (nz_block_idx[idx] != nz_block_idx[idx - 1]) {
                nz_block_idx_flag[idx] = nz_block_idx[idx];
            }
        }
    }
};
__global__ void CalcNzBlockIdxKernel(CalcNzBlockIdxWrapper op) {
    op();
}

struct SetNzFlagWrapper {
    bool* nz_flag;

    int nz_idx_num;
    int* nz_block_idx;
    int node_num;

    __device__ void operator()() {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < nz_idx_num) {
            int index                           = nz_block_idx[idx];
            int seri_i                          = index / node_num;
            int seri_j                          = index - seri_i * node_num;
            nz_flag[index]                      = true;
            nz_flag[seri_j * node_num + seri_i] = true;
        }
    }
};

__global__ void SetNzFlagKernel(SetNzFlagWrapper snf_wrapper) {
    snf_wrapper();
}

struct GetNnzPreWrapper {
    int* pre_nz_block_num;
    int* row_offset;
    bool* nz_flag;
    int node_num;

    // 每个block包含一个warp， 通过warp内协作进行prefix_sum
    __device__ void operator()() {
        __shared__ int s_pre[1];
        if (threadIdx.x == 0) {
            s_pre[0]                                = 0;
            pre_nz_block_num[node_num * blockIdx.x] = 0;
        }
        // __syncwarp();

        bool* flag_ptr = nz_flag + node_num * blockIdx.x;
        // nnz_pre_ptr需要的是exclude scan, 加一个偏移在include scan上实现
        int* nnz_pre_ptr = pre_nz_block_num + node_num * blockIdx.x + 1;
        int value, lane_id;

        // 循环内为warp内include scan（prefix sum），blockDim.x=32
        for (int iter = threadIdx.x; iter < node_num - 1; iter += blockDim.x) {
            lane_id = threadIdx.x & 0x1f;
            value   = flag_ptr[iter];
            for (int i = 1; i <= 32; i *= 2) {
                int n = __shfl_up_sync(__activemask(), value, i, 32);
                if ((lane_id & 31) >= i) {
                    value += n;
                }
            }
            nnz_pre_ptr[iter] = s_pre[0] + value;
            if (threadIdx.x == blockDim.x - 1) {
                s_pre[0] += value;
            }
            __syncwarp();
        }

        if (threadIdx.x == 0) {
            if (flag_ptr[node_num - 1]) {
                row_offset[blockIdx.x] = nnz_pre_ptr[node_num - 2] + 1;
            } else {
                row_offset[blockIdx.x] = nnz_pre_ptr[node_num - 2];
            }
        }
    }
};

__global__ void GetNnzPreKernel(GetNnzPreWrapper gnp_wrapper) {
    gnp_wrapper();
}

struct CalcPixelIdxNumForEachNzBlockWrapper {
    int* pixel_idx_num_vec;

    int nz_idx_num;
    int pixel_num;
    int* function_idx_vec;
    int* offset_vec;

    __device__ void operator()() {
        __shared__ int* s_func_list_ptr;
        __shared__ int s_num_fun;
        __shared__ int s_res_each_warp[4];   // for 4 warps

        if (blockIdx.x < nz_idx_num) {
            if (threadIdx.x == 0) {
                s_func_list_ptr = function_idx_vec + offset_vec[blockIdx.x];
                s_num_fun = offset_vec[blockIdx.x + 1] - offset_vec[blockIdx.x];
            }
            if (threadIdx.x < 4) {
                s_res_each_warp[threadIdx.x] = 0;
            }
            __syncthreads();

            bool is_pixel_idx;
            unsigned vote_result;
            int warp_id = threadIdx.x / 32;
            // warp内每个线程根据自己的数据（is_data_term）进行投票，然后统计投票结果。
            for (int iter = threadIdx.x; iter < s_num_fun; iter += blockDim.x) {
                is_pixel_idx = (s_func_list_ptr[iter] < pixel_num);
                vote_result  = __ballot_sync(__activemask(), is_pixel_idx);
                // first thread of each warp
                if (threadIdx.x == warp_id * 32 && vote_result > 0) {
                    // 这里vote_result的1一定是连续，且1在前0在后，小端模式存储，比如在内存中为11110000...
                    s_res_each_warp[warp_id] +=
                        int(log2f(float(vote_result) + 1.f));
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                pixel_idx_num_vec[blockIdx.x] =
                    s_res_each_warp[0] + s_res_each_warp[1] +
                    s_res_each_warp[2] + s_res_each_warp[3];
            }
        }
    }
};
__global__ void
CalcPixelIdxNumForEachNzBlockKernel(CalcPixelIdxNumForEachNzBlockWrapper op) {
    op();
}

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

    thrust::stable_sort_by_key(d_nz_block_idx_vec_duplicate.begin(),
                               d_nz_block_idx_vec_duplicate.end(),
                               d_function_idx_vec_.begin());

    /** 计算d_offset_vec_，nz_idx_num_ */
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
    thrust::remove(d_temp_array.begin() + 1, d_temp_array.end(), 0);
    d_nz_block_idx_vec_.resize(nz_block_num_);
    checkCudaErrors(
        cudaMemcpy(RAW_PTR(d_nz_block_idx_vec_), RAW_PTR(d_temp_array),
                   nz_block_num_ * sizeof(int), cudaMemcpyDeviceToDevice));

    /** 计算d_pixel_idx_num_vec_ */
    d_pixel_idx_num_vec_.resize(nz_block_num_, 0);

    CalcPixelIdxNumForEachNzBlockWrapper calc_pixel_idx_num_wrapper;
    calc_pixel_idx_num_wrapper.pixel_idx_num_vec =
        RAW_PTR(d_pixel_idx_num_vec_);
    calc_pixel_idx_num_wrapper.nz_idx_num       = nz_block_num_;
    calc_pixel_idx_num_wrapper.pixel_num        = pixel_num;
    calc_pixel_idx_num_wrapper.function_idx_vec = RAW_PTR(d_function_idx_vec_);
    calc_pixel_idx_num_wrapper.offset_vec       = RAW_PTR(d_offset_vec_);
    block = 32 * 4, grid = nz_block_num_;   // 每个block包含128个线程，4个warp
    CalcPixelIdxNumForEachNzBlockKernel<<<grid, block>>>(
        calc_pixel_idx_num_wrapper);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 第二步：将d_nz_block_idx_vec所表达的稀疏矩阵转换为CSR格式
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

    /** 计算d_pre_nz_idx_num_vec_和d_row_offset_vec_ */
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

    thrust::inclusive_scan(d_row_offset_vec.begin(), d_row_offset_vec.end(),
                           d_row_offset_vec_.begin() + 1);
}
