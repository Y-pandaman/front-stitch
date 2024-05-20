//
// Created by vradmin on 18-7-13.
//

#include "image_alignment_gpu.cuh"
#include "reduce.cuh"
#include "util/cuda_utils.h"
#include "util/innoreal_timer.hpp"
#include "util/math_utils.h"
// #include "test_JTJ.h"
#include "x_gn_solver.cuh"
#include <opencv2/imgproc/types_c.h>

inline float GetPixelValueBilinear(cv::Mat& img, float x, float y) {
    int x_0 = (int)x;
    int y_0 = (int)y;
    int x_1 = x_0 + 1;
    int y_1 = y_0 + 1;

    if (x_0 < 0 || y_0 < 0 || x_1 >= img.cols || y_1 >= img.rows)
        return 65535.0f;   // invalid

    float coef_x_0 = x_1 - x;
    float coef_y_0 = y_1 - y;

    float coef_00 = coef_x_0 * coef_y_0;
    float coef_10 = (1 - coef_x_0) * coef_y_0;
    float coef_01 = coef_x_0 * (1 - coef_y_0);
    float coef_11 = (1 - coef_x_0) * (1 - coef_y_0);

    float val_00 = img.at<float>(y_0, x_0);
    if (val_00 > 2.0f || val_00 < -2.0f)
        return 65535.0f;
    float val_10 = img.at<float>(y_0, x_1);
    if (val_10 > 2.0f || val_10 < -2.0f)
        return 65535.0f;
    float val_01 = img.at<float>(y_1, x_0);
    if (val_01 > 2.0f || val_01 < -2.0f)
        return 65535.0f;
    float val_11 = img.at<float>(y_1, x_1);
    if (val_11 > 2.0f || val_11 < -2.0f)
        return 65535.0f;

    return coef_00 * val_00 + coef_01 * val_01 + coef_10 * val_10 +
           coef_11 * val_11;
}

/**
 * ImageAlignmentCUDA的构造函数
 * 用于初始化图像对齐的CUDA处理。
 *
 * @param img_rows 图像的行数。
 * @param img_cols 图像的列数。
 * @param grid_num_rows 划分图像的网格行数。
 * @param grid_num_cols 划分图像的网格列数。
 */
ImageAlignmentCUDA::ImageAlignmentCUDA(int img_rows, int img_cols,
                                       int grid_num_rows, int grid_num_cols)
    : img_rows_(img_rows), img_cols_(img_cols),
      node_img_rows_(grid_num_rows + 1), node_img_cols_(grid_num_cols + 1) {
    // 初始化CUDA环境
    cudaFree(0);

    // 创建高斯滤波器
    gauss_filter_ = cv::cuda::createGaussianFilter(
        CV_32F, CV_32F, cv::Size(7, 7), 2.0, 0, cv::BORDER_DEFAULT, -1);
    cv::cuda::GpuMat temp(7, 7, CV_32FC1);
    gauss_filter_->apply(temp, temp);

    // 创建Scharr梯度滤波器
    // 创建水平和垂直方向的Scharr导数核
    cv::Mat kx_horizontal, ky_horizontal, kx_vertical, ky_vertical;
    cv::getDerivKernels(kx_horizontal, ky_horizontal, 1, 0, CV_SCHARR, true);
    cv::getDerivKernels(kx_vertical, ky_vertical, 0, 1, CV_SCHARR, true);

    // 使用Scharr导数核创建可分离的线性滤波器
    dx_scharr_filter_ = cv::cuda::createSeparableLinearFilter(
        CV_32F, CV_32F, kx_horizontal, ky_horizontal);
    dy_scharr_filter_ = cv::cuda::createSeparableLinearFilter(
        CV_32F, CV_32F, kx_vertical, ky_vertical);

    // 应用水平和垂直方向的导数滤波
    dx_scharr_filter_->apply(temp, temp);   // 对temp应用水平方向的导数滤波
    dy_scharr_filter_->apply(temp, temp);   // 对temp应用垂直方向的导数滤波

    // 计算网格大小和节点位置
    int grid_width  = (img_cols_ + grid_num_cols - 1) / grid_num_cols;
    int grid_height = (img_rows_ + grid_num_rows - 1) / grid_num_rows;

    // 预计算图像节点位置和像素到节点的映射
    std::vector<float2> node_vec;
    for (int row = 0, node_pos_row = 0; row < node_img_rows_;
         ++row, node_pos_row += grid_height) {
        for (int col = 0, node_pos_col = 0; col < node_img_cols_;
             ++col, node_pos_col += grid_width) {
            node_vec.push_back(make_float2(node_pos_col, node_pos_row));
        }
    }
    // 分配CUDA内存并复制节点向量
    checkCudaErrors(cudaMalloc(&d_node_vec_, node_vec.size() * sizeof(float2)));
    checkCudaErrors(
        cudaMalloc(&d_original_node_vec_, node_vec.size() * sizeof(float2)));
    checkCudaErrors(cudaMemcpy(d_original_node_vec_, node_vec.data(),
                               node_vec.size() * sizeof(float2),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_node_vec_, d_original_node_vec_,
                               node_vec.size() * sizeof(float2),
                               cudaMemcpyDeviceToDevice));

    // 预计算像素到节点的权重
    std::vector<int> pixel_rela_idx_vec;
    std::vector<float> pixel_rela_weight_vec;
    std::vector<int> node_rela_idx_vec;
    std::vector<float> node_rela_weight_vec;
    float coef_00, coef_10, coef_01, coef_11;
    for (int row = 0; row < img_rows; ++row) {
        for (int col = 0; col < img_cols; ++col) {
            int node_row = row / grid_height;
            int node_col = col / grid_width;
            // 像素到节点的索引和权重
            pixel_rela_idx_vec.push_back(node_row * node_img_cols_ + node_col);
            pixel_rela_idx_vec.push_back(node_row * node_img_cols_ + node_col +
                                         1);
            pixel_rela_idx_vec.push_back((node_row + 1) * node_img_cols_ +
                                         node_col);
            pixel_rela_idx_vec.push_back((node_row + 1) * node_img_cols_ +
                                         node_col + 1);
            CalcBilinearCoefs(coef_00, coef_10, coef_01, coef_11, col, row,
                              node_col * grid_width, node_row * grid_height,
                              (node_col + 1) * grid_width,
                              (node_row + 1) * grid_height);
            pixel_rela_weight_vec.push_back(coef_00);   // top-left
            pixel_rela_weight_vec.push_back(coef_10);   // top-right
            pixel_rela_weight_vec.push_back(coef_01);   // bottom-left
            pixel_rela_weight_vec.push_back(coef_11);   // bottom-right
        }
    }

    // 预计算节点之间的关系和权重
    for (int row = 0; row < node_img_rows_ - 1; ++row) {
        for (int col = 0; col < node_img_cols_ - 1; ++col) {
            node_rela_idx_vec.push_back(row * node_img_cols_ + col);
            node_rela_idx_vec.push_back(row * node_img_cols_ + col + 1);
            node_rela_idx_vec.push_back((row + 1) * node_img_cols_ + col + 1);
            node_rela_weight_vec.push_back(1.0f);
        }
    }
    for (int row = 1; row < node_img_rows_; ++row) {
        for (int col = 1; col < node_img_cols_; ++col) {
            node_rela_idx_vec.push_back(row * node_img_cols_ + col);
            node_rela_idx_vec.push_back(row * node_img_cols_ + col - 1);
            node_rela_idx_vec.push_back((row - 1) * node_img_cols_ + col - 1);
            node_rela_weight_vec.push_back(1.0f);
        }
    }

    // 分配CUDA内存并复制像素和节点的关系及权重向量
    checkCudaErrors(cudaMalloc(&d_pixel_rela_idx_vec_,
                               pixel_rela_idx_vec.size() * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_pixel_rela_weight_vec_,
                               pixel_rela_weight_vec.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_node_rela_idx_vec_,
                               node_rela_idx_vec.size() * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_node_rela_weight_vec_,
                               node_rela_weight_vec.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_pixel_rela_idx_vec_, pixel_rela_idx_vec.data(),
                               pixel_rela_idx_vec.size() * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_pixel_rela_weight_vec_, pixel_rela_weight_vec.data(),
        pixel_rela_weight_vec.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_node_rela_idx_vec_, node_rela_idx_vec.data(),
                               node_rela_idx_vec.size() * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(
        d_node_rela_weight_vec_, node_rela_weight_vec.data(),
        node_rela_weight_vec.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 初始化三角形参数
    triangle_u_ = 0.0f;
    triangle_v_ = -grid_width / (float)grid_height;

    // 初始化CUDA流
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams_[i]);
    }
    streams_array_.resize(4);

    // 设置求解器参数并初始化GNSolver
    const SolverPara para(1.0f, true, 0.4f, true, 0.005, true);
    gn_solver_ = new GNSolver(
        img_rows_, img_cols_, node_img_rows_, node_img_cols_,
        d_pixel_rela_idx_vec_, d_pixel_rela_weight_vec_, d_node_rela_idx_vec_,
        d_node_rela_weight_vec_, d_original_node_vec_, d_node_vec_,
        d_src_gray_img_, d_target_gray_img_, d_dy_img_, d_dx_img_, triangle_u_,
        triangle_v_, para);
}

ImageAlignmentCUDA::~ImageAlignmentCUDA() { }

void ImageAlignmentCUDA::ShowNodeImg(
    std::vector<std::vector<float2>>& node_vec_vis) { }

void ImageAlignmentCUDA::ShowComparisonImg() {
    cv::Mat img_1, img_2;
    d_warped_src_gray_img_.download(img_1);
    d_target_gray_img_.download(img_2);

    cv::Mat diff_image = cv::Mat::zeros(img_rows_, img_cols_, CV_32FC1);
    for (int row = 0; row < diff_image.rows; ++row) {
        for (int col = 0; col < diff_image.cols; ++col) {
            if (img_1.at<float>(row, col) > 2.0)
                img_1.at<float>(row, col) = 0.0f;
            if (img_2.at<float>(row, col) > 2.0)
                img_2.at<float>(row, col) = 0.0f;
            diff_image.at<float>(row, col) =
                fabs(img_1.at<float>(row, col) - img_2.at<float>(row, col));
        }
    }
    cv::imshow("diff_image", diff_image * 1);
    cv::imshow("img_1", img_1 * 1);
    cv::imshow("img_2", img_2 * 1);
    cv::waitKey(0);
}

__global__ void WarpSrcImgKernel(cv::cuda::PtrStep<float> warped_src_img,
                                 cv::cuda::PtrStepSz<float> src_img,
                                 cv::cuda::PtrStepSz<float> target_img,
                                 int pixel_num, int img_rows, int img_cols,
                                 float2* node_vec, int* pixel_rela_idx_vec,
                                 float* pixel_rela_weight_vec) {
    int pixel_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (pixel_idx >= pixel_num) {
        return;
    }

    int row = pixel_idx / img_cols;
    int col = pixel_idx % img_cols;
    int tl  = 4 * pixel_idx;
    int tr  = 4 * pixel_idx + 1;
    int bl  = 4 * pixel_idx + 2;
    int br  = 4 * pixel_idx + 3;

    float2 warped_pixel =
        pixel_rela_weight_vec[tl] * node_vec[pixel_rela_idx_vec[tl]] +
        pixel_rela_weight_vec[tr] * node_vec[pixel_rela_idx_vec[tr]] +
        pixel_rela_weight_vec[bl] * node_vec[pixel_rela_idx_vec[bl]] +
        pixel_rela_weight_vec[br] * node_vec[pixel_rela_idx_vec[br]];

    float src_rgb =
        GetPixelValueBilinearDevice(src_img, warped_pixel.x, warped_pixel.y);
    warped_src_img.ptr(row)[col] = src_rgb;
}

/**
 * @brief
 * 根据给定的参数，使用卷积的方式将源图像及其掩码映射到一个新的图像及其掩码上。
 *
 * @param warped_src_img 映射后的源图像，为 uchar3 类型数组，其中包含 RGB
 * 三通道颜色信息。
 * @param warped_src_mask 映射后的源图像掩码，为 uchar
 * 类型数组，用于标识像素是否有效。
 * @param src_img 源图像，为 uchar3 类型数组，其中包含 RGB 三通道颜色信息。
 * @param src_mask 源图像掩码，为 uchar 类型数组，用于标识像素是否有效。
 * @param pixel_num 图像中像素的数量。
 * @param img_rows 图像的行数。
 * @param img_cols 图像的列数。
 * @param node_vec 包含节点信息的浮点型数组。
 * @param pixel_rela_idx_vec 包含像素相对索引的整型数组。
 * @param pixel_rela_weight_vec 包含像素相对权重的浮点型数组。
 */
__global__ void WarpSrcImgKernel(uchar3* warped_src_img, uchar* warped_src_mask,
                                 uchar3* src_img, uchar* src_mask,
                                 int pixel_num, int img_rows, int img_cols,
                                 float2* node_vec, int* pixel_rela_idx_vec,
                                 float* pixel_rela_weight_vec) {
    // 计算当前处理的像素索引
    int pixel_idx = threadIdx.x + blockDim.x * blockIdx.x;
    // 如果索引超出图像范围，则返回
    if (pixel_idx >= pixel_num) {
        return;
    }

    // 根据索引计算像素的行列位置
    int row = pixel_idx / img_cols;
    int col = pixel_idx % img_cols;
    // 计算四个角落像素在数组中的索引
    int tl = 4 * pixel_idx;
    int tr = 4 * pixel_idx + 1;
    int bl = 4 * pixel_idx + 2;
    int br = 4 * pixel_idx + 3;

    // 根据权重和节点信息计算当前像素的映射位置
    float2 warped_pixel =
        pixel_rela_weight_vec[tl] * node_vec[pixel_rela_idx_vec[tl]] +
        pixel_rela_weight_vec[tr] * node_vec[pixel_rela_idx_vec[tr]] +
        pixel_rela_weight_vec[bl] * node_vec[pixel_rela_idx_vec[bl]] +
        pixel_rela_weight_vec[br] * node_vec[pixel_rela_idx_vec[br]];

    // 计算映射后像素的四个相邻像素索引
    int x_0          = (int)warped_pixel.x;
    int y_0          = (int)warped_pixel.y;
    int x_1          = x_0 + 1;
    int y_1          = y_0 + 1;
    int pixel_idx_00 = y_0 * img_cols + x_0;
    int pixel_idx_10 = y_0 * img_cols + x_1;
    int pixel_idx_01 = y_1 * img_cols + x_0;
    int pixel_idx_11 = y_1 * img_cols + x_1;

    // 检查映射后的像素是否在图像范围内
    if (x_0 < 0 || y_0 < 0 || x_1 >= img_cols || y_1 >= img_rows) {
        // 如果不在范围内，则将像素设置为黑色，掩码设置为0
        warped_src_img[pixel_idx]  = make_uchar3(0, 0, 0);
        warped_src_mask[pixel_idx] = 0;
        return;
    }

    // 检查四个相邻像素是否都有效
    if (src_mask[pixel_idx_00] > 128 && src_mask[pixel_idx_10] > 128 &&
        src_mask[pixel_idx_01] > 128 && src_mask[pixel_idx_11] > 128) {
        // 如果都有效，则设置掩码为255
        warped_src_mask[pixel_idx] = 255;

        // 计算插值权重
        float coef_x_0 = x_1 - warped_pixel.x;
        float coef_y_0 = y_1 - warped_pixel.y;
        float coef_00  = coef_x_0 * coef_y_0;
        float coef_10  = (1 - coef_x_0) * coef_y_0;
        float coef_01  = coef_x_0 * (1 - coef_y_0);
        float coef_11  = (1 - coef_x_0) * (1 - coef_y_0);
        // 获取四个相邻像素的颜色值
        uchar3 val_00 = src_img[pixel_idx_00];
        uchar3 val_10 = src_img[pixel_idx_10];
        uchar3 val_01 = src_img[pixel_idx_01];
        uchar3 val_11 = src_img[pixel_idx_11];
        // 使用插值计算当前像素的颜色
        float3 rgb =
            coef_00 * make_float3((int)val_00.x, (int)val_00.y, (int)val_00.z) +
            coef_01 * make_float3((int)val_01.x, (int)val_01.y, (int)val_01.z) +
            coef_10 * make_float3((int)val_10.x, (int)val_10.y, (int)val_10.z) +
            coef_11 * make_float3((int)val_11.x, (int)val_11.y, (int)val_11.z);
        // 设置当前像素的颜色
        warped_src_img[pixel_idx] =
            make_uchar3((int)rgb.x, (int)rgb.y, (int)rgb.z);
    } else {
        // 如果不都有效，则将像素设置为黑色，掩码设置为0
        warped_src_img[pixel_idx]  = make_uchar3(0, 0, 0);
        warped_src_mask[pixel_idx] = 0;
    }
}

void ImageAlignmentCUDA::WarpSrcImg() {
    /** backward mapping */
    int pixel_num = img_rows_ * img_cols_;
    d_warped_src_gray_img_.create(img_rows_, img_cols_, CV_32FC1);
    int block = 128, grid = (pixel_num + block - 1) / block;
    WarpSrcImgKernel<<<grid, block>>>(
        d_warped_src_gray_img_, d_src_gray_img_, d_target_gray_img_, pixel_num,
        img_rows_, img_cols_, RAW_PTR(d_node_vec_),
        RAW_PTR(d_pixel_rela_idx_vec_), RAW_PTR(d_pixel_rela_weight_vec_));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 在GPU上运行的内核函数，用于设置图像中指定像素的无效值。
 *
 * @param img_float 指向图像数据（浮点数表示）的指针。
 * @param img_uchar3 指向图像数据（无符号字符三元组表示）的指针。
 * @param mask_uchar
 * 指向掩码数据（无符号字符表示）的指针，用于确定哪些像素需要被处理。
 * @param pixel_num 像素总数。
 *
 * 对于掩码中值大于128的像素，该函数将基于RGB值计算新的浮点数表示，并更新图像数据；
 * 对于掩码中值不大于128的像素，将其浮点数表示设置为一个特定的无效值。
 */
__global__ void SetInvalidValuesKernel(float* img_float, uchar3* img_uchar3,
                                       uchar* mask_uchar, int pixel_num) {
    // 根据块索引和线程索引计算当前处理的像素索引
    int pixel_idx = blockDim.x * blockIdx.x + threadIdx.x;

    // 如果当前像素索引超出图像像素总数，则函数提前终止
    if (pixel_idx >= pixel_num) {
        return;
    }

    // 检查当前像素是否被选中处理
    if (mask_uchar[pixel_idx] > 128) {
        // 如果像素被选中，根据RGB值计算新的浮点数表示
        uchar3 rgb = img_uchar3[pixel_idx];
        img_float[pixel_idx] =
            ((int)rgb.x + (int)rgb.y + (int)rgb.z) / 3.0f / 255.0f;
    } else {
        // 如果像素未被选中，设置为特定的无效值
        img_float[pixel_idx] = 65535.0f;
    }
}

void ImageAlignmentCUDA::SetSrcTargetImgsFromHost(
    cv::Mat& src_img, cv::Mat& src_mask, int2 src_min_uv, cv::Mat& target_img,
    cv::Mat& target_mask, int2 target_min_uv) {
    int src_pixel_num    = src_img.rows * src_img.cols;
    int target_pixel_num = target_img.rows * target_img.cols;

    uchar3 *d_src_img, *d_target_img;
    uchar *d_src_mask, *d_target_mask;
    checkCudaErrors(cudaMalloc(&d_src_img, src_pixel_num * sizeof(uchar3)));
    checkCudaErrors(
        cudaMalloc(&d_target_img, target_pixel_num * sizeof(uchar3)));
    checkCudaErrors(cudaMalloc(&d_src_mask, src_pixel_num * sizeof(uchar)));
    checkCudaErrors(
        cudaMalloc(&d_target_mask, target_pixel_num * sizeof(uchar)));
    checkCudaErrors(cudaMemcpy(d_src_img, src_img.data,
                               src_pixel_num * sizeof(uchar3),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_target_img, target_img.data,
                               target_pixel_num * sizeof(uchar3),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_src_mask, src_mask.data,
                               src_pixel_num * sizeof(uchar),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_target_mask, target_mask.data,
                               target_pixel_num * sizeof(uchar),
                               cudaMemcpyHostToDevice));

    float *src_temp_vec, *target_temp_vec;
    checkCudaErrors(cudaMalloc(&src_temp_vec, src_pixel_num * sizeof(float)));
    checkCudaErrors(
        cudaMalloc(&target_temp_vec, target_pixel_num * sizeof(float)));

    int block = 128, grid = (src_pixel_num + block - 1) / block;
    SetInvalidValuesKernel<<<grid, block>>>(src_temp_vec, (uchar3*)d_src_img,
                                            d_src_mask, src_pixel_num);
    grid = (target_pixel_num + block - 1) / block;
    SetInvalidValuesKernel<<<grid, block>>>(target_temp_vec,
                                            (uchar3*)d_target_img,
                                            d_target_mask, target_pixel_num);

    SetSrcTargetImgs(src_temp_vec, src_img.rows, src_img.cols, target_temp_vec,
                     target_img.rows, target_img.cols, src_min_uv,
                     target_min_uv);
}

/**
 * 将源图像和目标图像的数据从设备（GPU）设置到类的成员变量中。
 *
 * @param d_src_img 指向源图像（GPU内存）的uchar3类型指针。
 * @param d_src_mask 指向源图像掩码（GPU内存）的uchar类型指针。
 * @param src_rows 源图像的行数。
 * @param src_cols 源图像的列数。
 * @param d_target_img 指向目标图像（GPU内存）的uchar3类型指针。
 * @param d_target_mask 指向目标图像掩码（GPU内存）的uchar类型指针。
 * @param target_rows 目标图像的行数。
 * @param target_cols 目标图像的列数。
 *
 * 此函数首先确保源图像和目标图像的缓冲区已在GPU内存中分配，
 * 然后调用CUDA内核函数SetInvalidValuesKernel来处理图像数据，
 * 最后更新类的成员变量以指向处理后的图像数据。
 */
void ImageAlignmentCUDA::SetSrcTargetImgsFromDevice(
    uchar3* d_src_img, uchar* d_src_mask, int src_rows, int src_cols,
    uchar3* d_target_img, uchar* d_target_mask, int target_rows,
    int target_cols) {
    // 确保当前CUDA设备的操作已完成，并检查是否有错误发生。
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 计算源图像和目标图像的像素数量。
    int src_pixel_num    = src_rows * src_cols;
    int target_pixel_num = target_rows * target_cols;

    // 如果源图像和目标图像的缓冲区尚未分配，则进行分配。
    if (d_src_img_buffer_ == nullptr) {
        checkCudaErrors(
            cudaMalloc(&d_src_img_buffer_, src_pixel_num * sizeof(float)));
        checkCudaErrors(cudaMalloc(&d_target_img_buffer_,
                                   target_pixel_num * sizeof(float)));
    }

    // 使用块和网格维度设置无效值内核，并确保执行完毕且无错误。
    int block = 128, grid = (src_pixel_num + block - 1) / block;
    SetInvalidValuesKernel<<<grid, block>>>(
        d_src_img_buffer_, (uchar3*)d_src_img, d_src_mask, src_pixel_num);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 对目标图像执行相同的操作。
    grid = (target_pixel_num + block - 1) / block;
    SetInvalidValuesKernel<<<grid, block>>>(d_target_img_buffer_,
                                            (uchar3*)d_target_img,
                                            d_target_mask, target_pixel_num);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 更新源和目标图像的数据到类的成员变量。
    SetSrcTargetImgs(d_src_img_buffer_, src_rows, src_cols,
                     d_target_img_buffer_, target_rows, target_cols,
                     make_int2(0, 0), make_int2(0, 0));
}

/**
 * 在CUDA上执行图像变形操作，将源图像按照指定的映射关系变形。
 *
 * @param d_warped_src_img 指向变形后图像数据的指针（ uchar3
 * 类型，包含RGB三个通道）。
 * @param d_warped_src_mask 指向变形后图像掩码数据的指针（ uchar
 * 类型，1个通道）。
 * @param d_src_img 指向源图像数据的指针（ uchar3 类型，包含RGB三个通道）。
 * @param d_src_mask 指向源图像掩码数据的指针（ uchar 类型，1个通道）。
 * @param src_rows 源图像的行数。
 * @param src_cols 源图像的列数。
 *
 * 函数首先检查是否已经分配了图像缓冲区，如果没有，则分配足够的内存以存储源图像和掩码。
 * 接着，将源图像和掩码数据从设备内存拷贝到分配的缓冲区中。然后，通过调用 CUDA
 * 核函数 WarpSrcImgKernel 对源图像进行变形处理，将结果存储到 d_warped_src_img
 * 和 d_warped_src_mask 指向的设备内存中。
 */
void ImageAlignmentCUDA::WarpSrcImg(uchar3* d_warped_src_img,
                                    uchar* d_warped_src_mask, uchar3* d_src_img,
                                    uchar* d_src_mask, int src_rows,
                                    int src_cols) {
    // 检查图像缓冲区是否已分配，若未分配则进行分配
    if (image_buffer_ == nullptr) {
        checkCudaErrors(
            cudaMalloc(&image_buffer_, src_rows * src_cols * sizeof(uchar3)));
        checkCudaErrors(
            cudaMalloc(&mask_buffer_, src_rows * src_cols * sizeof(uchar)));
    }
    // 将源图像和掩码数据从设备内存拷贝到缓冲区
    checkCudaErrors(cudaMemcpy(image_buffer_, d_src_img,
                               src_rows * src_cols * sizeof(uchar3),
                               cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(mask_buffer_, d_src_mask,
                               src_rows * src_cols * sizeof(uchar),
                               cudaMemcpyDeviceToDevice));

    /** 进行图像的反向映射 */
    int pixel_num = img_rows_ * img_cols_;
    d_warped_src_gray_img_.create(img_rows_, img_cols_, CV_32FC1);
    // 计算执行kernel的网格和块的大小
    int block = 128, grid = (pixel_num + block - 1) / block;

    // 调用CUDA核函数进行图像变形
    WarpSrcImgKernel<<<grid, block>>>(
        d_warped_src_img, d_warped_src_mask, image_buffer_, mask_buffer_,
        pixel_num, img_rows_, img_cols_, RAW_PTR(d_node_vec_),
        RAW_PTR(d_pixel_rela_idx_vec_), RAW_PTR(d_pixel_rela_weight_vec_));
    // 等待所有CUDA任务完成，并检查是否有错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 设置源图像和目标图像，进行图像灰度化和滤波处理。
 *
 * @param d_src_img 指向源图像数据的设备端指针。
 * @param src_rows 源图像的行数。
 * @param src_cols 源图像的列数。
 * @param d_target_img 指向目标图像数据的设备端指针。
 * @param target_rows 目标图像的行数。
 * @param target_cols 目标图像的列数。
 * @param src_min_uv 源图像在UV坐标系中的最小坐标。
 * @param target_min_uv 目标图像在UV坐标系中的最小坐标。
 * @param gpu_idx 使用的GPU索引。
 */
void ImageAlignmentCUDA::SetSrcTargetImgs(float* d_src_img, int src_rows,
                                          int src_cols, float* d_target_img,
                                          int target_rows, int target_cols,
                                          int2 src_min_uv, int2 target_min_uv,
                                          int gpu_idx) {
    src_min_uv_    = src_min_uv;
    target_min_uv_ = target_min_uv;

    // 如果图像矩阵未初始化，则创建新的矩阵
    if (d_src_gray_img_.rows == 0) {
        d_src_gray_img_.create(src_rows, src_cols, CV_32FC1);
        d_target_gray_img_.create(target_rows, target_cols, CV_32FC1);
    }
    // 将源图像和目标图像数据从设备复制到设备
    checkCudaErrors(cudaMemcpy2D(d_src_gray_img_.data, d_src_gray_img_.step,
                                 d_src_img, src_cols * sizeof(float),
                                 src_cols * sizeof(float), src_rows,
                                 cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy2D(
        d_target_gray_img_.data, d_target_gray_img_.step, d_target_img,
        src_cols * sizeof(float), target_cols * sizeof(float), target_rows,
        cudaMemcpyDeviceToDevice));

    // 对源图像和目标图像应用高斯滤波
    gauss_filter_->apply(d_src_gray_img_, d_src_gray_img_);
    gauss_filter_->apply(d_target_gray_img_, d_target_gray_img_);
    // 对源图像应用Scharr梯度滤波
    dy_scharr_filter_->apply(d_src_gray_img_, d_dy_img_);
    dx_scharr_filter_->apply(d_src_gray_img_, d_dx_img_);
}

/** node建立在target图上，warp的时候使用backward mapping */
void ImageAlignmentCUDA::AlignSrcImgToTarget(int iter_num) {
    for (int iter = 0; iter < iter_num; ++iter) {
        gn_solver_->Next();
    }
}
