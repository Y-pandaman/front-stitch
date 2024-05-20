#include "util/cuda_utils.h"
#include "util/helper_cuda.h"
#include "x_pcg_solver.cuh"
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>

struct DotMuliplyWrapper {
    int num_data;
    const float* input_data1;
    const float* input_data2;
    float* output;
    /**
     * 在CUDA设备上执行的运算符函数。
     * 该函数用于将两个输入数据数组的元素逐个相乘，并将结果存储到输出数组中。
     * 这个函数是为并行计算设计的，每个线程处理数组中的一个元素。
     *
     * 参数列表：
     * 无参数，但函数体内部使用了全局变量 input_data1, input_data2, output,
     * num_data。 这些变量应在函数外部定义和初始化。
     *
     * 返回值：
     * 无返回值。
     */
    __device__ void operator()() {
        // 计算当前线程处理的数据索引
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        // 如果索引在有效数据范围内，则进行乘法运算并存储结果
        if (idx < num_data) {
            output[idx] = input_data1[idx] * input_data2[idx];
        }
    }
};
/**
 * 在GPU上执行点乘运算的内核函数。
 *
 * @param dm_wrapper 一个DotMuliplyWrapper类型的对象，封装了点乘运算的具体实现。
 *                   通过此参数将计算逻辑传递到内核函数中执行。
 *
 * 该函数通过调用DotMuliplyWrapper对象的函数调用操作符()，在GPU的线程中执行点乘运算。
 *_dot_
 */
__global__ void DotMuliplyKernel(DotMuliplyWrapper dm_wrapper) {
    // 调用DotMuliplyWrapper对象，执行点乘运算
    dm_wrapper();
}

struct InnerProductWrapper {
    int num_data;
    const float* input_data1;
    const float* input_data2;
    float* output;
    /**
     * 在CUDA设备上执行的函数，用于对两个输入数据数组进行逐元素乘积的累加，然后将结果归约到一个单一的输出值。
     * 这个函数是为了解决在单个
     * warp（一组32个线程）内进行数据点的快速矢量乘法和累加而设计的。
     *
     * 该函数不接受参数，也不返回值，但要求作为全局变量或外部参数提供的有：
     * - `input_data1` 和 `input_data2`：两个输入数据数组，其长度由 `num_data`
     * 指定。
     * - `num_data`：输入数据数组的长度。
     * - `output`：用于存储计算结果的外部浮点数变量。
     *
     * 注意：此函数假设每个warp处理的数据是独立的，且最终将所有warp的结果归约到一个单一的输出值。
     */
    __device__ void operator()() {
        __shared__ float s_res[32];   // 分配共享内存以存储每个warp的结果

        int warp_id = threadIdx.x / 32;   // 计算当前线程所属的warp ID

        // 初始化每个warp的结果存储
        if (threadIdx.x - warp_id * 32 == 31) {
            s_res[warp_id] = 0;
        }

        float value = 0.f;   // 初始化线程本地的累加器
        // 遍历数据并计算累加值
        for (int iter = threadIdx.x; iter < num_data; iter += blockDim.x) {
            value += input_data1[iter] * input_data2[iter];
        }

        // 使用SHFL_XOR指令进行warp内的归约
        for (int i = 16; i >= 1; i /= 2) {
            value += __shfl_xor_sync(__activemask(), value, i, 32);
        }

        // 将warp的结果写入共享内存
        if (threadIdx.x - warp_id * 32 == 31) {
            s_res[warp_id] = value;
        }
        __syncthreads();   // 确保所有线程都完成了它们的操作

        // 将所有warp的结果归约到一个单一的值
        if (threadIdx.x == 0) {
            float res_out = 0.f;
            for (int iter = 0; iter < 32; ++iter) {
                res_out += s_res[iter];
            }
            output[0] = res_out;   // 将最终结果写入全局输出变量
        }
    }
};
/**
 * 在GPU上执行内积计算的内核函数。
 *
 * @param ip_wrapper
 * 一个封装了内积计算的对象，其类型为InnerProductWrapper。该对象必须能被调用（如通过operator()）以执行实际的内积计算。
 *
 * 该函数将调用传入的ip_wrapper对象，从而在GPU的多个线程中并行执行内积计算。
 */
__global__ void InnerProductKernel(InnerProductWrapper ip_wrapper) {
    // 在每个线程上调用ip_wrapper对象以执行内积计算的一部分。
    ip_wrapper();
}

/**
 * @brief 更新alpha值的内核函数
 *
 * 该函数在GPU上运行，用于更新数组d_alpha的第一个元素。它通过将d_zr_dot的第一个元素除以d_alpha的第一个元素来完成更新。
 *
 * @param d_alpha 指向GPU内存中float类型数组的指针，该数组的第一个元素将被更新。
 * @param d_zr_dot
 * 指向GPU内存中float类型数组的指针，该数组的第一个元素用于更新d_alpha的第一个元素。
 */
__global__ void UpdateAlphaKernel(float* d_alpha, float* d_zr_dot) {
    // 只有线程索引为0的线程执行更新操作
    if (threadIdx.x == 0) {
        d_alpha[0] = d_zr_dot[0] / d_alpha[0];
    }
}

/**
 * AxpyOperatorWrapper 结构体定义了一个用于执行AXPY操作的封装器。
 * AXPY（Alpha * X + Y）是一个线性代数操作，其中Alpha是标量，X和Y是向量。
 * 这个封装器旨在被CUDA内核调用，用于并行地对向量Y进行更新。
 */
struct AxpyOperatorWrapper {
    int num_data;            // 数据的数量，即向量x和y的长度。
    const float* scale_a;    // 用于缩放X向量的标量a的指针。
    const float* vector_x;   // 待操作的输入向量x的指针。
    float* vector_y;         // 待更新的输出向量y的指针。

    /**
     * 这个成员函数定义了CUDA内核的操作。
     * 它被设计为在GPU上并行执行AXPY操作。
     */
    __device__ void operator()() {
        int idx = threadIdx.x +
                  blockDim.x * blockIdx.x;   // 计算当前线程处理的数据索引。
        if (idx < num_data) {
            vector_y[idx] +=
                scale_a[0] *
                vector_x[idx];   // 如果索引在有效范围内，则更新向量y。
        }
    }
};

/**
 * 在GPU上执行Axpy操作的内核函数。
 *
 * @param ao_wrapper
 * Axpy操作包装器对象，是一个可调用对象，用于在GPU线程中执行具体的Axpy操作。
 *                   此参数通过设备内存传递，并在内核函数中被调用以执行操作。
 *
 * 该函数设计为在CUDA并行计算模型下运行，每个线程调用一次ao_wrapper()，从而实现对大型数据集的并行处理。
 */
__global__ void AxpyOperatorKernel(AxpyOperatorWrapper ao_wrapper) {
    // 在每个线程中调用Axpy操作包装器对象
    ao_wrapper();
}

/**
 * 结构体 UpdateZAndRWrapper 用于封装并行计算过程中所需的数据和操作。
 * 该结构体主要用于更新变量 z 和 r，其中涉及到了一些预处理步骤。
 *
 * @param num_data 指明数据的数量。
 * @param omega 存储权重参数的数组。
 * @param preconditioner 预处理因子数组。
 * @param scale_a 缩放因子数组，目前只使用了第一个元素。
 * @param d_r 存储 r 数据的设备端数组。
 * @param d_z 存储 z 数据的设备端数组。
 */
struct UpdateZAndRWrapper {
    int num_data;
    const float* omega;
    const float* preconditioner;
    const float* scale_a;
    float* d_r;
    float* d_z;

    /**
     * __device__ 标记的成员函数表明该函数可以在 GPU 线程中调用。
     * 函数作用是更新 d_r 和 d_z 数组中的值，基于输入的 omega、preconditioner 和
     * scale_a。 对于每个线程块，线程将并行处理一部分数据。
     */
    __device__ void operator()() {
        int idx = threadIdx.x +
                  blockDim.x * blockIdx.x;   // 计算当前线程处理的数据索引
        if (idx < num_data) {   // 检查索引是否在有效范围内，防止越界
            float res_r = d_r[idx] - scale_a[0] * omega[idx];   // 更新 r 的值
            d_r[idx]    = res_r;   // 将计算结果写回 d_r
            d_z[idx] =
                preconditioner[idx] * res_r;   // 使用预处理因子更新 z 的值
        }
    }
};
/**
 * 在GPU上执行的更新Z和R的内核函数。
 *
 * @param uzar_wrapper
 * 一个封装了更新Z和R逻辑的可调用对象。此对象应包含所有必要的数据和操作，
 *                     以在GPU的线程中执行更新逻辑。
 *
 * 该函数的设计目的是通过调用传入的UpdateZAndRWrapper对象的函数调用操作，来更新Z和R的值。
 * 由于这是一个内核函数，它将在GPU的多个线程中并发执行，每个线程都会调用uzar_wrapper对象，
 * 从而实现对Z和R的并行更新。
 */
__global__ void UpdateZAndRKernel(UpdateZAndRWrapper uzar_wrapper) {
    uzar_wrapper();   // 调用UpdateZAndRWrapper对象，执行Z和R的更新逻辑
}

/**
 * @brief 更新beta值的核函数
 *
 * 该函数在GPU上执行，用于根据当前和旧的zr_dot值更新beta值。
 *
 * @param d_zr_dot 指向当前zr_dot值的设备内存指针。
 * @param d_zr_dot_old 指向旧的zr_dot值的设备内存指针。
 * @param d_beta 指向需要更新的beta值的设备内存指针。
 */
__global__ void UpdataBetaKernel(float* d_zr_dot, float* d_zr_dot_old,
                                 float* d_beta) {
    // 只有线程0负责计算和更新beta值
    if (threadIdx.x == 0) {
        d_beta[0] = d_zr_dot[0] / d_zr_dot_old[0];
    }
}

/**
 * XpayOperatorWrapper 结构体定义了一个用于执行向量运算的 CUDA 函数。
 * 该结构体通过成员变量保存了运算所需的数据规模、缩放因子、输入向量和输出向量。
 * 其中，成员函数 `operator()` 定义了在 CUDA 设备上执行的具体运算逻辑。
 */
struct XpayOperatorWrapper {
    int num_data;           // 数据数量
    const float* scale_a;   // 缩放因子的指针，目前只支持一个缩放因子
    const float* vector_x;   // 输入向量x的指针
    float* vector_y;         // 输出向量y的指针

    /**
     * CUDA 设备上的成员函数，用于执行具体的向量运算。
     * 每个线程处理一个数据元素，将 vector_y 中的元素乘以 scale_a 并与 vector_x
     * 中对应元素相加， 然后将结果保存回 vector_y。
     */
    __device__ void operator()() {
        int idx = threadIdx.x +
                  blockDim.x * blockIdx.x;   // 计算当前线程处理的数据索引
        if (idx < num_data) {
            vector_y[idx] = scale_a[0] * vector_y[idx] +
                            vector_x[idx];   // 执行缩放和相加操作
        }
    }
};
/**
 * 在GPU上执行Xapy操作器的内核函数。
 *
 * @param xo_wrapper 一个封装了Xapy操作的可调用对象，此对象将在GPU的线程中执行。
 *
 * 该函数设计为在CUDA并行计算模型下运行，每个线程将调用xo_wrapper执行具体操作。
 */
__global__ void XapyOperatorKernel(XpayOperatorWrapper xo_wrapper) {
    // 在这里，我们简单地调用xo_wrapper对象，执行封装的操作。
    xo_wrapper();
}

struct SparseMvWrapper {
    const int* ia;
    const int* ja;
    const float* a;
    const float* x;
    float* res;
    /**
     * 在CUDA的块级别上对64个线程的数据进行缩减操作。
     * 使用此函数，每个块可以将64个线程（每个线程一个数据）的累加结果减少为一个值。
     *
     * @param s_tmp_res 一个共享内存数组，用于存储每个 warp 的缩减结果。
     * @param s_data 输入数据的共享内存数组，每个线程读取一个数据。
     * @param res 输出结果的指针，存储块级别的缩减结果。
     */
    __device__ void BlockReduce64(float* s_tmp_res, const float* s_data,
                                  float* res) {
        // 计算当前线程所属的warp ID
        int warp_id = threadIdx.x >> 5;
        // 初始化存储缩减结果的共享内存
        if (threadIdx.x < 2) {
            s_tmp_res[threadIdx.x] = 0.f;
        }

        // 获取当前线程处理的数据
        float value = s_data[threadIdx.x];
        // 使用shfl_xor指令进行warp内的数据聚合
        for (int i = 16; i >= 1; i >>= 1) {
            value += __shfl_xor_sync(__activemask(), value, i, 32);
        }
        // 将warp的缩减结果写入共享内存
        if ((threadIdx.x & 0x1f) == 31) {
            s_tmp_res[warp_id] = value;
        }
        __syncthreads();   // 确保所有线程都完成写入

        // 将所有warp的缩减结果累加得到最终的块级缩减结果
        if (threadIdx.x == 0) {
            *res = s_tmp_res[0] + s_tmp_res[1];
        }
    }

    /**
     * 在设备上执行的函数，用于对稀疏矩阵和向量的乘法进行局部求和，然后通过块级归约计算最终结果。
     * 这个函数不接受参数，也不返回值，但会通过共享内存和块级归约来计算并更新全局结果数组的一部分。
     *
     * 具体操作包括：
     * 1. 从全局内存读取稀疏矩阵的行起始索引和非零元素数量。
     * 2. 每个线程负责对矩阵的一行进行乘法累加操作。
     * 3. 使用块级归约技术对线程内的结果进行归约。
     */
    __device__ void operator()() {
        __shared__ float s_res_tmp[64];   // 共享内存用于存储线程局部结果
        __shared__ int s_start_idx;   // 当前行在稀疏矩阵中的起始索引
        __shared__ int s_num_row_nnz;   // 当前行中的非零元素数量
        __shared__ float
            s_reduce_tmp[2];   // 共享内存用于存储块级归约的中间结果

        // 初始化共享内存，读取稀疏矩阵当前行的起始索引和非零元素数量
        if (threadIdx.x == 0) {
            s_start_idx   = ia[blockIdx.x];
            s_num_row_nnz = ia[blockIdx.x + 1] - s_start_idx;
        }
        __syncthreads();   // 确保所有线程都读取到了正确的值

        s_res_tmp[threadIdx.x] = 0.f;   // 初始化线程局部结果

        // 循环处理当前行的所有非零元素，进行乘法累加
        for (int iter = threadIdx.x; iter < s_num_row_nnz; iter += blockDim.x) {
            s_res_tmp[threadIdx.x] +=
                a[s_start_idx + iter] * x[ja[s_start_idx + iter]];
        }
        __syncthreads();   // 等待所有线程完成累加操作

        // 使用块级归约计算最终结果，并存储到全局内存中
        BlockReduce64(s_reduce_tmp, s_res_tmp, &(res[blockIdx.x]));
    }
};
/**
 * @brief 在GPU上执行稀疏矩阵向量乘法的内核函数。
 *
 * @param spmv_wrapper 稀疏矩阵向量乘法的包装器对象，封装了实际的计算逻辑。
 *                     该参数被设计为一个可调用对象，以在内核函数中直接调用。
 */
__global__ void SparseMvKernel(SparseMvWrapper spmv_wrapper) {
    // 在这里，我们利用线程块和网格的概念，将计算分配给大量的线程。
    // 每个线程将处理矩阵中的一部分元素与向量的乘法加法操作。
    spmv_wrapper();   // 直接调用传递的包装器对象，执行稀疏矩阵向量乘法的具体计算。
}

/**
 * PcgLinearSolverGPU的构造函数
 * 用于初始化PCG线性解算器的GPU版本。
 *
 * @param row 稀疏矩阵的行数，决定了向量的大小。
 */
PcgLinearSolverGPU::PcgLinearSolverGPU(int row)
    : cusparseHandle_(nullptr), descr_(nullptr), row_(row) {
    // 创建cusparse句柄和矩阵描述符
    cusparseCreate(&cusparseHandle_);
    cusparseCreateMatDescr(&descr_);
    // 设置矩阵描述符的索引基为0
    cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);

    // 分配GPU缓冲区，用于存储迭代过程中的临时向量
    d_buf_.resize(row_ * 5);
    // 从缓冲区中定义向量的起始指针
    d_omega_   = RAW_PTR(d_buf_);
    d_p_       = RAW_PTR(d_buf_) + row_;
    d_r_       = RAW_PTR(d_buf_) + row_ * 2;
    d_z_       = RAW_PTR(d_buf_) + row_ * 3;
    d_delta_x_ = RAW_PTR(d_buf_) + row_ * 4;

    // 分配GPU内存用于存储特定计算（如点乘、内积）的结果
    d_zr_dot_.resize(1);
    d_zr_dot_old_.resize(1);
    d_alpha_.resize(1);
    d_beta_.resize(1);

    // 创建并初始化各种操作器和包装器类的实例，用于执行PCG算法中的各种操作
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

/**
 * 在GPU上使用PCG方法解决线性方程组
 *
 * @param d_x 指向解向量的设备指针
 * @param d_ia 指向稀疏矩阵非零元素行起点的设备指针
 * @param d_ja 指向稀疏矩阵非零元素列索引的设备指针
 * @param d_a 指向稀疏矩阵非零元素值的设备指针
 * @param num_nnz 稀疏矩阵的非零元素数量
 * @param d_b 指向右侧向量的设备指针
 * @param precond 预条件器的设备指针
 * @param max_iter 最大迭代次数
 */
void PcgLinearSolverGPU::Solve(float* d_x, int* d_ia, int* d_ja, float* d_a,
                               int num_nnz, float* d_b, float* precond,
                               int max_iter) {
    // 初始化残差向量r为b
    checkCudaErrors(cudaMemcpy(RAW_PTR(d_r_), RAW_PTR(d_b),
                               sizeof(float) * row_, cudaMemcpyDeviceToDevice));

    // 设置CUDA块和网格大小
    int block_size = 256;
    int grid_size  = (row_ + block_size - 1) / block_size;

    // 初始化数据结构以进行点乘运算
    dm_wrapper_->num_data    = row_;
    dm_wrapper_->input_data1 = RAW_PTR(precond);
    dm_wrapper_->input_data2 = RAW_PTR(d_r_);
    dm_wrapper_->output      = RAW_PTR(d_p_);
    DotMuliplyKernel<<<grid_size, block_size>>>(*dm_wrapper_);
    // 确保所有操作都已完成，检查错误
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 设置稀疏矩阵-向量乘法(SPMV)的参数
    spmv_wrapper_->ia  = d_ia;
    spmv_wrapper_->ja  = d_ja;
    spmv_wrapper_->a   = d_a;
    spmv_wrapper_->x   = RAW_PTR(d_p_);
    spmv_wrapper_->res = RAW_PTR(d_omega_);

    // 初始化内积计算的相关结构
    ip_wrapper_->num_data    = row_;
    ip_wrapper_->input_data1 = RAW_PTR(d_r_);
    ip_wrapper_->input_data2 = RAW_PTR(d_p_);
    ip_wrapper_->output      = RAW_PTR(d_zr_dot_);
    InnerProductKernel<<<1, 1024>>>(*ip_wrapper_);
    // 确保所有操作都已完成，检查错误
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 后续步骤中用到的Alpha计算设置
    ip_wrapper_alpha_->num_data    = row_;
    ip_wrapper_alpha_->input_data1 = RAW_PTR(d_p_);
    ip_wrapper_alpha_->input_data2 = RAW_PTR(d_omega_);
    ip_wrapper_alpha_->output      = RAW_PTR(d_alpha_);

    // 设置Axpy操作的相关参数
    ao_wrapper_->num_data = row_;
    ao_wrapper_->vector_x = RAW_PTR(d_p_);
    ao_wrapper_->vector_y = RAW_PTR(d_x);
    ao_wrapper_->scale_a  = RAW_PTR(d_alpha_);

    // 设置Uzawa迭代的相关参数
    uzar_wrapper_->num_data       = row_;
    uzar_wrapper_->preconditioner = RAW_PTR(precond);
    uzar_wrapper_->omega          = RAW_PTR(d_omega_);
    uzar_wrapper_->d_r            = RAW_PTR(d_r_);
    uzar_wrapper_->d_z            = RAW_PTR(d_z_);
    uzar_wrapper_->scale_a        = RAW_PTR(d_alpha_);

    // 设置Beta计算的相关参数
    ip_wrapper_beta_->num_data    = row_;
    ip_wrapper_beta_->input_data1 = RAW_PTR(d_z_);
    ip_wrapper_beta_->input_data2 = RAW_PTR(d_r_);
    ip_wrapper_beta_->output      = RAW_PTR(d_zr_dot_);

    // 设置Xapy操作的相关参数
    xo_wrapper_->num_data = row_;
    xo_wrapper_->vector_x = RAW_PTR(d_z_);
    xo_wrapper_->vector_y = RAW_PTR(d_p_);
    xo_wrapper_->scale_a  = RAW_PTR(d_beta_);

    // 主循环：直到残差足够小或达到最大迭代次数
    float zr_dot;
    for (int k = 0; k < max_iter; ++k) {
        // 保存当前的zr_dot以用于Beta的更新
        checkCudaErrors(cudaMemcpy(RAW_PTR(d_zr_dot_old_), RAW_PTR(d_zr_dot_),
                                   sizeof(float), cudaMemcpyDeviceToDevice));
        // 将zr_dot拷贝到主机以供检查
        checkCudaErrors(cudaMemcpy(&zr_dot, RAW_PTR(d_zr_dot_), sizeof(float),
                                   cudaMemcpyDeviceToHost));
        // 如果残差足够小，则终止迭代
        if (zr_dot < 1.0e-5f)
            break;

        // 执行稀疏矩阵-向量乘法(SPMV)
        SparseMvKernel<<<row_, 64>>>(*spmv_wrapper_);

        // 更新Alpha值
        InnerProductKernel<<<1, 1024>>>(*ip_wrapper_alpha_);
        UpdateAlphaKernel<<<1, 1>>>(RAW_PTR(d_alpha_), RAW_PTR(d_zr_dot_));

        // 使用Alpha更新解向量x
        AxpyOperatorKernel<<<grid_size, block_size>>>(*ao_wrapper_);

        // 更新r和z向量
        UpdateZAndRKernel<<<grid_size, block_size>>>(*uzar_wrapper_);

        // 更新Beta值
        InnerProductKernel<<<1, 1024>>>(*ip_wrapper_beta_);
        UpdataBetaKernel<<<1, 1>>>(RAW_PTR(d_zr_dot_), RAW_PTR(d_zr_dot_old_),
                                   RAW_PTR(d_beta_));

        // 使用Beta更新p向量
        XapyOperatorKernel<<<grid_size, block_size>>>(*xo_wrapper_);
    }
    // 确保所有操作都已完成，检查错误
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}
