#include "common/cylinder_stitcher.cuh"
#include "common/cylinder_stitcher.h"
#include "common/multiband_blend.cuh"
#include "common/render.cuh"
#include "render/image_alignment_gpu.cuh"
#include "render/project_to_cylinder.cuh"
#include "render/seam_finder.cuh"
#include "util/cuda_utils.h"
#include "util/innoreal_timer.hpp"
#include "util/math_utils.h"
#include <Eigen/Eigen>
#include <fstream>
#include <opencv2/opencv.hpp>

#define PI 3.14159265358979323846
#define INF 10000000.0f

using namespace cv;
using namespace cv::detail;

/**
 * CylinderStitcherGPU的构造函数
 * 用于初始化GPU上用于拼接圆柱形图像的相关资源。
 *
 * @param view_num 视图数量，表示需要拼接的图像数量。
 * @param buffer_size 缓冲区大小，以像素为单位，用于存储图像数据。
 */
CylinderStitcherGPU::CylinderStitcherGPU(int view_num, int buffer_size) {
    view_num_ = view_num;

    // 分配GPU内存用于存储图像数据及其他辅助信息
    checkCudaErrors(cudaMalloc(
        (void**)&d_buffer_1_,
        sizeof(uchar3) * buffer_size));   // 用于存储图像像素的临时缓冲区（RGB）

    checkCudaErrors(cudaMalloc(
        (void**)&d_buffer_2_,
        sizeof(uchar) * buffer_size));   // 可能用于存储单通道图像数据的缓冲区

    checkCudaErrors(cudaMalloc(
        (void**)&d_buffer_3_,
        sizeof(uchar3) *
            buffer_size));   // 另一个用于存储图像像素的临时缓冲区（RGB）

    checkCudaErrors(cudaMalloc(
        (void**)&d_buffer_4channels_,
        sizeof(uchar4) *
            buffer_size));   // 用于存储四通道图像数据的缓冲区（RGBA）

    checkCudaErrors(
        cudaMalloc((void**)&bounding_boxes_image_,
                   sizeof(uchar3) *
                       buffer_size));   // 用于存储边界框图像数据的缓冲区（RGB）

    views_.resize(view_num);        // 为每个视图准备空间
    cyl_images_.resize(view_num);   // 为拼接后的圆柱形图像准备空间
}

/**
 * 将图像数据设置到GPU的缓冲区中，为后续的处理做准备。
 *
 * @param images 指向图像数据的指针，图像数据为三维像素格式(uchar3)。
 * @param width 图像的宽度。
 * @param height 图像的高度。
 * @param view_num 视图的数量，即图像的数量。
 */
void CylinderStitcherGPU::setImages(uchar3* images, int width, int height,
                                    int view_num) {
    // 将图像数据从主机内存拷贝到GPU的d_buffer_1_缓冲区
    checkCudaErrors(cudaMemcpy(d_buffer_1_, images,
                               width * height * view_num * sizeof(uchar3),
                               cudaMemcpyHostToDevice));
    // 更新每个视图的图像指针，指向d_buffer_1_中相应视图的图像数据开始处
    for (int i = 0; i < view_num; ++i) {
        views_[i].image = (uchar3*)(&(d_buffer_1_[i * width * height * 3]));
    }
}

__global__ void ConvertRGBAF2RGBU(float* img_rgba, uchar3* img_rgb, int width,
                                  int height) {
    int pixel_idx = blockDim.x * blockIdx.x + threadIdx.x;

    int max_idx = height * width;
    if (pixel_idx >= max_idx)
        return;

    int dst_row = pixel_idx / width, dst_col = pixel_idx % width;
    int src_row = height - dst_row - 1, src_col = dst_col;
    int src_pixel_idx    = (src_row * width + src_col) * 4;
    img_rgb[pixel_idx].x = img_rgba[src_pixel_idx] * 255;
    img_rgb[pixel_idx].y = img_rgba[src_pixel_idx + 1] * 255;
    img_rgb[pixel_idx].z = img_rgba[src_pixel_idx + 2] * 255;
}

/**
 * @brief 将RGBA浮点数图像转换为RGBA无符号字符图像。
 *
 * 此函数在CUDA的并行计算模型下运行，将源图像（以浮点数形式的RGBA格式存储）转换为目标图像（以无符号字符四元数RGBA格式存储）。
 * 主要进行颜色空间的转换，并同时处理图像的翻转（垂直翻转）。
 *
 * @param src_img_rgba 指向源图像（RGBA浮点数格式）数据的指针。
 * @param tgt_img_rgba 指向目标图像（RGBA无符号字符四元数格式）数据的指针。
 * @param width 图像的宽度。
 * @param height 图像的高度。
 */
__global__ void ConvertRGBAF2RGBAU(float* src_img_rgba, uchar4* tgt_img_rgba,
                                   int width, int height) {
    // 根据线程索引计算当前处理的像素索引
    int pixel_idx = blockDim.x * blockIdx.x + threadIdx.x;

    // 计算处理所有像素的最大索引值
    int max_idx = height * width;
    // 如果当前像素索引超出图像范围，则返回
    if (pixel_idx >= max_idx)
        return;

    // 计算目标图像中的行和列索引
    int dst_row = pixel_idx / width, dst_col = pixel_idx % width;
    // 计算源图像中的行和列索引，实现图像的垂直翻转
    int src_row = height - dst_row - 1, src_col = dst_col;
    // 根据源图像的行、列索引计算像素在源图像中的索引
    int src_pixel_idx = (src_row * width + src_col) * 4;
    // 将源图像的RGBA值转换并赋值给目标图像，同时进行归一化到[0, 255]
    tgt_img_rgba[pixel_idx].x = src_img_rgba[src_pixel_idx] * 255;
    tgt_img_rgba[pixel_idx].y = src_img_rgba[src_pixel_idx + 1] * 255;
    tgt_img_rgba[pixel_idx].z = src_img_rgba[src_pixel_idx + 2] * 255;
    tgt_img_rgba[pixel_idx].w = src_img_rgba[src_pixel_idx + 3] * 255;
}

/**
 * 将额外的图像数据设置到GPU上，用于后续的处理。
 *
 * @param image_cuda 指向GPU上存放的额外图像数据的指针。
 * @param width 图像的宽度。
 * @param height 图像的高度。
 *
 * 此函数首先更新额外视图的尺寸信息，然后将传入的RGBA浮点图像转换为RGBA无符号字符图像，
 * 存储到类成员变量d_buffer_4channels_中，供其他GPU方法使用。
 */
void CylinderStitcherGPU::setExtraImageCuda(float* image_cuda, int width,
                                            int height) {
    // 更新额外视图的尺寸信息，mask设为nullptr
    extra_view_.width  = width;
    extra_view_.height = height;
    extra_view_.mask   = nullptr;

    // 计算CUDA kernel调用的网格和块大小
    int block = 128, grid = (height * width + block - 1) / block;

    // 确保之前的操作已经完成
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 将RGBA浮点图像转换为RGBA无符号字符图像
    ConvertRGBAF2RGBAU<<<grid, block>>>(
        image_cuda, (uchar4*)d_buffer_4channels_, width, height);

    // 确保kernel调用和后续操作完成
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 更新额外视图的图像指针
    extra_view_.image = (uchar4*)d_buffer_4channels_;
}

void CylinderStitcherGPU::setBoundingBoxesImage(uchar3* image, int width,
                                                int height) {
    checkCudaErrors(cudaMemcpy(bounding_boxes_image_, image,
                               width * height * sizeof(uchar3),
                               cudaMemcpyHostToDevice));
}

/**
 * 将一系列图像的掩码设置到GPU缓冲区。
 * @param images 指向包含所有视图掩码的uchar数组的指针。
 * @param width 图像的宽度。
 * @param height 图像的高度。
 * @param view_num 视图的数量。
 */
void CylinderStitcherGPU::setMasks(uchar* images, int width, int height,
                                   int view_num) {
    // 将主机上的图像掩码数据复制到GPU的d_buffer_2_上
    checkCudaErrors(cudaMemcpy(d_buffer_2_, images,
                               width * height * view_num * sizeof(uchar),
                               cudaMemcpyHostToDevice));
    // 遍历所有视图，设置每个视图的掩码指针、宽度和高度
    for (int i = 0; i < view_num; ++i) {
        views_[i].mask   = (uchar*)(&(d_buffer_2_[i * width * height]));
        views_[i].width  = width;
        views_[i].height = height;
    }
}

/**
 * 设置相机参数
 * 本函数用于设置视图的内部和外部参数，为每个视图创建一个针孔相机模型，并将其存储在CylinderStitcherGPU的成员变量views_中。
 *
 * @param intrins 视图的内部参数向量，每个向量包含4个浮点数，分别代表相机的fx,
 * fy, cx, cy。
 * @param distorts
 * 视图的畸变参数向量，每个向量包含4个浮点数，用于描述相机的径向和切向畸变。
 * @param extrins
 * 视图的外部参数向量，包含所有视图的位姿信息。每个位姿是一个4x4的变换矩阵，描述了相机在全局坐标系中的位置和朝向。
 */
void CylinderStitcherGPU::setCameras(std::vector<float4> intrins,
                                     std::vector<float4> distorts,
                                     std::vector<float> extrins) {
    // 将外部参数矩阵转换为相机位姿，并存储在camera_poses中
    std::vector<Eigen::Matrix4f> camera_poses;
    for (int i = 0; i < view_num_; ++i) {
        Eigen::Matrix4f extrin, camera_pose;
        // 构建相机的外部参数矩阵
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                extrin(row, col) = extrins[16 * i + row * 4 + col];
            }
        }
        // 计算相机位姿，从外部参数矩阵求逆
        camera_poses.emplace_back(extrin.inverse());
    }

    // 遍历所有视图，为每个视图设置相机模型
    for (int i = 0; i < view_num_; ++i) {
        Eigen::Matrix4f camera_pose = camera_poses[i];
        Eigen::Matrix4f extrin      = camera_pose.inverse();
        std::vector<float> R, T, C;
        float *d_R, *d_T, *d_C;

        // 提取旋转矩阵的值并传输到GPU内存
        R.emplace_back(extrin(0, 0));
        R.emplace_back(extrin(0, 1));
        R.emplace_back(extrin(0, 2));
        R.emplace_back(extrin(1, 0));
        R.emplace_back(extrin(1, 1));
        R.emplace_back(extrin(1, 2));
        R.emplace_back(extrin(2, 0));
        R.emplace_back(extrin(2, 1));
        R.emplace_back(extrin(2, 2));
        checkCudaErrors(cudaMalloc((void**)&d_R, 9 * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_R, R.data(), 9 * sizeof(float),
                                   cudaMemcpyHostToDevice));

        // 提取平移向量的值并传输到GPU内存
        T.emplace_back(extrin(0, 3));
        T.emplace_back(extrin(1, 3));
        T.emplace_back(extrin(2, 3));
        checkCudaErrors(cudaMalloc((void**)&d_T, 3 * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_T, T.data(), 3 * sizeof(float),
                                   cudaMemcpyHostToDevice));

        // 提取相机中心的值并传输到GPU内存
        C.emplace_back(camera_pose(0, 3));
        C.emplace_back(camera_pose(1, 3));
        C.emplace_back(camera_pose(2, 3));
        checkCudaErrors(cudaMalloc((void**)&d_C, 3 * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_C, C.data(), 3 * sizeof(float),
                                   cudaMemcpyHostToDevice));

        // 创建针孔相机模型并赋值给每个视图
        views_[i].camera =
            PinholeCameraGPU(intrins[i].x, intrins[i].y, intrins[i].z,
                             intrins[i].w, distorts[i].x, distorts[i].y,
                             distorts[i].z, distorts[i].w, d_R, d_T, d_C);
    }
}

/**
 * 设置额外视图相机参数
 * 本函数用于根据给定的内参和外参设置额外视图的相机参数。其中，内参以float4形式提供，外参以一维float向量形式提供。
 *
 * @param intrin float4类型的内参，依次为fx, fy, cx,
 * cy，其中fx和fy分别为图像的横纵方向像素大小，cx和cy为图像的光心坐标。
 * @param extrin_float
 * 一维float向量，包含外参矩阵的元素。外参矩阵为4x4的齐次变换矩阵，此处以一维向量形式传入，需要通过代码还原矩阵。
 */
void CylinderStitcherGPU::setExtraViewCamera(float4 intrin,
                                             std::vector<float> extrin_float) {
    // 将外参向量转换为4x4矩阵并求其逆矩阵，得到相机位姿矩阵camera_pose
    Eigen::Matrix4f extrin, camera_pose;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            extrin(row, col) = extrin_float[row * 4 + col];
        }
    }
    camera_pose = extrin.inverse();

    // 分别准备旋转矩阵R、平移向量T和相机中心C的GPU内存，并将它们从主机内存拷贝到GPU内存
    std::vector<float> R, T, C;
    float *d_R, *d_T, *d_C;

    // 提取旋转矩阵R的元素并分配GPU内存，准备进行CUDA内存拷贝
    R.emplace_back(extrin(0, 0));
    R.emplace_back(extrin(0, 1));
    R.emplace_back(extrin(0, 2));
    R.emplace_back(extrin(1, 0));
    R.emplace_back(extrin(1, 1));
    R.emplace_back(extrin(1, 2));
    R.emplace_back(extrin(2, 0));
    R.emplace_back(extrin(2, 1));
    R.emplace_back(extrin(2, 2));
    checkCudaErrors(cudaMalloc((void**)&d_R, 9 * sizeof(float)));
    checkCudaErrors(
        cudaMemcpy(d_R, R.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));

    // 提取平移向量T的元素并分配GPU内存，准备进行CUDA内存拷贝
    T.emplace_back(extrin(0, 3));
    T.emplace_back(extrin(1, 3));
    T.emplace_back(extrin(2, 3));
    checkCudaErrors(cudaMalloc((void**)&d_T, 3 * sizeof(float)));
    checkCudaErrors(
        cudaMemcpy(d_T, T.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));

    // 提取相机中心C的元素并分配GPU内存，准备进行CUDA内存拷贝
    C.emplace_back(camera_pose(0, 3));
    C.emplace_back(camera_pose(1, 3));
    C.emplace_back(camera_pose(2, 3));
    checkCudaErrors(cudaMalloc((void**)&d_C, 3 * sizeof(float)));
    checkCudaErrors(
        cudaMemcpy(d_C, C.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));

    // 使用提供的内参和提取的外参参数初始化额外视图的相机对象
    extra_view_.camera = PinholeCameraGPU(intrin.x, intrin.y, intrin.z,
                                          intrin.w, 0, 0, 0, 0, d_R, d_T, d_C);
}

/**
 * 对左右两个图像进行对齐操作。
 * 此函数首先设置图像对齐器的源图像和目标图像，然后执行图像对齐，
 * 最后对源图像进行变形，使其与目标图像对齐。
 *
 * @param time 一个用于控制迭代次数的参数。如果时间为0，迭代次数为30；否则为4。
 */
void CylinderStitcherGPU::alignImages(int time) {
    // 设置左侧图像对齐器的源图像和目标图像
    image_alignment_left_->SetSrcTargetImgsFromDevice(
        cyl_images_[0].image, cyl_images_[0].mask, cyl_images_[0].height,
        cyl_images_[0].width, cyl_images_[1].image, cyl_images_[1].mask,
        cyl_images_[1].height, cyl_images_[1].width);
    // 设置右侧图像对齐器的源图像和目标图像
    image_alignment_right_->SetSrcTargetImgsFromDevice(
        cyl_images_[2].image, cyl_images_[2].mask, cyl_images_[2].height,
        cyl_images_[2].width, cyl_images_[1].image, cyl_images_[1].mask,
        cyl_images_[1].height, cyl_images_[1].width);

    // 根据时间参数确定迭代次数
    int iter_num = 4;
    if (time == 0)
        iter_num = 30;
    // 对左侧图像进行对齐
    image_alignment_left_->AlignSrcImgToTarget(iter_num);
    // 对右侧图像进行对齐
    image_alignment_right_->AlignSrcImgToTarget(iter_num);

    // 对左侧图像进行变形，使其与目标图像对齐
    image_alignment_left_->WarpSrcImg(
        cyl_images_[0].image, cyl_images_[0].mask, cyl_images_[0].image,
        cyl_images_[0].mask, cyl_images_[0].height, cyl_images_[0].width);
    // 对右侧图像进行变形，使其与目标图像对齐
    image_alignment_right_->WarpSrcImg(
        cyl_images_[2].image, cyl_images_[2].mask, cyl_images_[2].image,
        cyl_images_[2].mask, cyl_images_[2].height, cyl_images_[2].width);
}

/**
 * 在CylinderStitcherGPU类中，查找并计算图像的接缝。
 * 该方法首先检查seam_masks_是否为空，如果为空，则初始化seam_masks_，
 * 并为左右两条线分配CUDA内存。之后，调用SeamFind_cuda函数来查找接缝。
 * 如果seam_masks_不为空，则只调用SeamFind_cuda函数进行接缝查找，不进行初始化。
 */
void CylinderStitcherGPU::findSeam() {
    // 当seam_masks_为空时，初始化seam_masks_并为每条线分配内存
    if (seam_masks_.size() == 0) {
        separate_lines_.resize(2);   // 初始化分离线数组

        int width  = cyl_images_[1].width;    // 获取图像宽度
        int height = cyl_images_[1].height;   // 获取图像高度

        int interval = width / 3;   // 计算分离线之间的间隔
        // 为每行添加分离线位置
        for (int i = 0; i < height; i++) {
            separate_lines_[0].emplace_back(interval);
            separate_lines_[1].emplace_back(2 * interval);
        }

        // 分配CUDA内存以存储接缝掩码
        uchar *seam_mask_left, *seam_mask_mid, *seam_mask_right;
        cudaMalloc((void**)&seam_mask_left, sizeof(uchar) * height * width);
        cudaMalloc((void**)&seam_mask_mid, sizeof(uchar) * height * width);
        cudaMalloc((void**)&seam_mask_right, sizeof(uchar) * height * width);

        // 将接缝掩码指针添加到seam_masks_中
        seam_masks_.emplace_back(seam_mask_left);
        seam_masks_.emplace_back(seam_mask_mid);
        seam_masks_.emplace_back(seam_mask_right);

        // 调用CUDA函数进行接缝查找
        SeamFind_cuda(cyl_images_, cyl_images_[1].height, cyl_images_[1].width,
                      separate_lines_, 150, seam_masks_, true);
    } else {
        // 如果seam_masks_不为空，则只进行接缝查找
        SeamFind_cuda(cyl_images_, cyl_images_[1].height, cyl_images_[1].width,
                      separate_lines_, 20, seam_masks_, false);
    }
}

/**
 * 获取图像mask的边界点位置
 *
 * 本函数遍历输入的mask图像，对于mask中非零的像素点，检查其是否位于图像的边界，
 * 或者其相邻像素是否为零。如果是，则将该像素点的位置添加到边界位置列表中。
 *
 * @param mask 输入的mask图像，类型为cv::Mat，要求为8位单通道图像。
 * @param boundary_loc
 * 存储边界点位置的向量，每个边界点用int2类型表示，其中x坐标为列号，y坐标为行号。
 */
static void GetBoundary(cv::Mat& mask, std::vector<int2>& boundary_loc) {
    // 遍历mask图像的所有像素
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            uchar m = mask.at<uchar>(i, j);   // 获取当前像素值
            if (m == 0)
                continue;   // 如果当前像素为0，则跳过处理

            // 检查当前像素是否位于图像边界
            if (i == 0 || j == 0 || i == mask.rows - 1 || j == mask.cols - 1) {
                boundary_loc.emplace_back(make_int2(
                    j, i));   // 如果是边界像素，则添加到boundary_loc中
            } else {
                // 检查当前像素的相邻像素是否为0
                if (mask.at<uchar>(i - 1, j) == 0 ||
                    mask.at<uchar>(i, j - 1) == 0 ||
                    mask.at<uchar>(i + 1, j) == 0 ||
                    mask.at<uchar>(i, j + 1) == 0) {
                    boundary_loc.emplace_back(make_int2(
                        j, i));   // 如果有相邻像素为0，则添加到boundary_loc中
                }
            }
        }
    }
}

static bool RayCylinderIntersection(Eigen::Matrix<float, 3, 1> origin,
                                    Eigen::Matrix<float, 3, 1> dir, float r,
                                    Eigen::Matrix<float, 3, 1>& intersection) {
    if (origin.norm() > r) {
        std::cout << "Outside the cylinder" << std::endl;
        return false;
    }

    float k, b;
    float intersect_x, intersect_z;

    if (dir(0, 0) == 0 && dir(0, 0) == 0)
        return false;

    if (std::fabs(dir(0, 0)) < 1.0e-6) {
        intersect_x = origin(0, 0);
        float z1    = std::sqrt(r * r - intersect_x * intersect_x);
        float z2    = -z1;

        if ((z1 - origin(2, 0)) / dir(2, 0) > 0) {
            intersect_z = z1;
        } else {
            intersect_z = z2;
        }
    } else if (std::fabs(dir(2, 0)) < 1.0e-6) {
        intersect_z = origin(2, 0);
        float x1    = std::sqrt(r * r - intersect_z * intersect_z);
        float x2    = -x1;

        if ((x1 - origin(0, 0)) / dir(0, 0) > 0) {
            intersect_x = x1;
        } else {
            intersect_x = x2;
        }
    } else {
        k = dir(2, 0) / dir(0, 0);
        b = origin(2, 0) - k * origin(0, 0);
        float check =
            std::pow(2 * k * b, 2.0f) - 4 * (1 + k * k) * (b * b - r * r);
        if (check < 0)
            return false;
        float item = std::sqrt(check);

        float x1 = (-2.0f * k * b + item) / (2 * (1 + k * k));
        float x2 = (-2.0f * k * b - item) / (2 * (1 + k * k));

        float z1 = k * x1 + b;
        float z2 = k * x2 + b;

        if ((x1 - origin(0, 0)) / dir(0, 0) > 0 &&
            (z1 - origin(2, 0)) / dir(2, 0) > 0) {
            intersect_x = x1;
            intersect_z = z1;
        } else {
            intersect_x = x2;
            intersect_z = z2;
        }
    }

    float t_xz = std::sqrt(std::pow(intersect_x - origin(0, 0), 2.0f) +
                           std::pow(intersect_z - origin(2, 0), 2.0f));
    float tan_theta =
        dir(1, 0) / std::sqrt(dir(0, 0) * dir(0, 0) + dir(2, 0) * dir(2, 0));

    float t_y         = t_xz * tan_theta;
    float intersect_y = origin(1, 0) + t_y;

    intersection << intersect_x, intersect_y, intersect_z;

    return true;
}

static bool BackProj(float theta, float phi, float& x, float& y,
                     PinholeCamera* cam, Cylinder* cyl) {
    Eigen::Matrix<float, 3, 1> P;
    P << cyl->r * std::sin(theta), phi * cyl->r, cyl->r * std::cos(theta);

    Eigen::Matrix<float, 3, 1> target_pixel =
        cam->K * cam->R *
        (cyl->rotation.transpose() * P + cyl->center - cam->C);
    if (target_pixel(2, 0) <= 0)
        return false;

    x = target_pixel(0, 0) / target_pixel(2, 0);
    y = target_pixel(1, 0) / target_pixel(2, 0);
    return true;
}

static bool ForwardProjToCylinder(int x, int y, float& theta, float& phi,
                                  PinholeCamera* cam, Cylinder* cyl) {
    Eigen::Matrix<float, 3, 1> P;
    Ray ray = cam->getRay(x, y);
    Eigen::Matrix<float, 3, 1> ray_o =
        cyl->rotation * (ray.origin - cyl->center);
    Eigen::Matrix<float, 3, 1> ray_dir = cyl->rotation * ray.dir;
    bool success = RayCylinderIntersection(ray_o, ray_dir, cyl->r, P);

    if (!success) {
        std::cout << "Ray Cylinder intersect failed! " << std::endl;
        return false;
    }

    theta = clamp(atan2f(P(0, 0), P(2, 0)), -3.141592653 / 2 + 0.001,
                  3.141592653 / 2 - 0.001);
    phi   = P(1, 0) / cyl->r;

    return true;
}

bool Merge(float2& merged_tl, float2& merged_rb, float2 tl_i, float2 rb_i,
           float2 tl_j, float2 rb_j) {
    if ((tl_i.x > tl_j.x && tl_i.x < rb_j.x && tl_i.y > tl_j.y &&
         tl_i.y < rb_j.y) ||
        (rb_i.x > tl_j.x && rb_i.x < rb_j.x && rb_i.y > tl_j.y &&
         rb_i.y < rb_j.y) ||
        (tl_j.x > tl_i.x && tl_j.x < rb_i.x && tl_j.y > tl_i.y &&
         tl_j.y < rb_i.y) ||
        (rb_j.x > tl_i.x && rb_j.x < rb_i.x && rb_j.y > tl_i.y &&
         rb_j.y < rb_i.y)) {
        merged_tl.x = min(tl_i.x, tl_j.x);
        merged_tl.y = min(tl_i.y, tl_j.y);
        merged_rb.x = min(rb_i.x, rb_j.x);
        merged_rb.y = min(rb_i.y, rb_j.y);

        return true;
    }
    return false;
}

void CylinderStitcherGPU::drawBoundingBoxes(cv::Mat& image,
                                            std::vector<float2>& boxes) {
    int width  = cyl_images_[cyl_images_.size() / 2].width;
    int height = cyl_images_[cyl_images_.size() / 2].height;
    image      = cv::Mat::zeros(height, width, CV_8UC3);
    for (int i = 0; i < boxes.size() / 2; ++i) {
        float2 tl = boxes[2 * i];
        float2 rb = boxes[2 * i + 1];
        cv::Rect2f rec(tl.x, tl.y, rb.x - tl.x, rb.y - tl.y);
        cv::rectangle(image, rec, cv::Scalar(0, 0, 255), -1);
    }
}

void CylinderStitcherGPU::drawBoundingBoxes_2(cv::Mat& image,
                                              std::vector<float2>& boxes) {
    int width  = cyl_images_[cyl_images_.size() / 2].width;
    int height = cyl_images_[cyl_images_.size() / 2].height;
    image      = cv::Mat::zeros(height, width, CV_8UC3);
    for (int i = 0; i < boxes.size() / 2; ++i) {
        float2 tl = boxes[2 * i];
        float2 rb = boxes[2 * i + 1];
    }
    {
        float global_min_theta, global_max_theta, global_min_phi,
            global_max_phi;
        cudaMemcpy(&global_min_theta, cyl_->global_theta, sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&global_max_theta, cyl_->global_theta + 1, sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&global_min_phi, cyl_->global_phi, sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&global_max_phi, cyl_->global_phi + 1, sizeof(float),
                   cudaMemcpyDeviceToHost);

        int width    = cyl_images_[cyl_images_.size() / 2].width;
        int height   = cyl_images_[cyl_images_.size() / 2].height;
        float step_x = ((global_max_theta) - (global_min_theta)) / width;
        float step_y = ((global_max_phi) - (global_min_phi)) / height;

        std::vector<float> rotation(9);
        cudaMemcpy(rotation.data(), cyl_->rotation, sizeof(float) * 9,
                   cudaMemcpyDeviceToHost);
        h_cyl_.rotation << rotation[0], rotation[1], rotation[2], rotation[3],
            rotation[4], rotation[5], rotation[6], rotation[7], rotation[8];

        std::vector<float> center(3);
        cudaMemcpy(center.data(), cyl_->center, sizeof(float) * 3,
                   cudaMemcpyDeviceToHost);
        h_cyl_.center << center[0], center[1], center[2];
        h_cyl_.r = cyl_->r;

        std::vector<float> R_vec(9), T_vec(3), C_vec(3);
        cudaMemcpy(R_vec.data(), extra_view_.camera.R, sizeof(float) * 9,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(T_vec.data(), extra_view_.camera.T, sizeof(float) * 3,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(C_vec.data(), extra_view_.camera.C, sizeof(float) * 3,
                   cudaMemcpyDeviceToHost);

        PinholeCamera camera;
        camera.K << extra_view_.camera.fx, 0, extra_view_.camera.cx, 0,
            extra_view_.camera.fy, extra_view_.camera.cy, 0, 0, 1;
        camera.R << R_vec[0], R_vec[1], R_vec[2], R_vec[3], R_vec[4], R_vec[5],
            R_vec[6], R_vec[7], R_vec[8];
        camera.T << T_vec[0], T_vec[1], T_vec[2];
        camera.C << C_vec[0], C_vec[1], C_vec[2];

        for (int box_idx = 0; box_idx < boxes.size() / 2; ++box_idx) {
            float2& tl = boxes[2 * box_idx];
            float2& rb = boxes[2 * box_idx + 1];
            float theta, phi;
            float2 tl_xy, rb_xy;
            phi = tl.y * step_y + global_min_phi;
            ;
            theta = tl.x * step_x + global_min_theta;
            BackProj(theta, phi, tl_xy.x, tl_xy.y, &camera, &h_cyl_);
            phi = rb.y * step_y + global_min_phi;
            ;
            theta = rb.x * step_x + global_min_theta;
            BackProj(theta, phi, rb_xy.x, rb_xy.y, &camera, &h_cyl_);

            cv::Rect2f rec(tl_xy.x, tl_xy.y, rb_xy.x - tl_xy.x,
                           rb_xy.y - tl_xy.y);
            cv::rectangle(image, rec, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    }
}

/**
 * 将当前视图投影到圆柱形图像上
 *
 * @param time 当前时间或帧数，用于可能的时间依赖处理，本函数未直接使用此参数
 */
void CylinderStitcherGPU::stitch_project_to_cyn(int time) {
    // 如果圆柱体对象未初始化，则进行初始化
    if (cyl_ == nullptr) {
        // 初始化圆柱体参数
        float r = 1000.0;
        cyl_    = new CylinderGPU(views_[1].camera.R, views_[1].camera.C, r);

        // 遍历所有视图，获取视图边界位置
        std::vector<std::vector<int2>> boundary_locs;
        for (int i = 0; i < views_.size(); ++i) {
            cv::Mat h_mask(views_[i].height, views_[i].width, CV_8UC1);
            cudaMemcpy(h_mask.data, views_[i].mask,
                       sizeof(uchar) * views_[i].height * views_[i].width,
                       cudaMemcpyDeviceToHost);
            std::vector<int2> boundary_loc;
            GetBoundary(h_mask, boundary_loc);   // 获取视图边界点
            boundary_locs.emplace_back(boundary_loc);
        }

        // 设置圆柱体的边界位置
        cyl_->setBoundary(boundary_locs);

        // 计算圆柱图像的宽度和高度
        cyl_image_width_  = views_[1].width * 3 / 2;
        cyl_image_height_ = views_[1].height * 1.2;
        // 调整图像尺寸以适应网格数
        cyl_image_width_ =
            ((cyl_image_width_ - 1) / col_grid_num_ + 1) * col_grid_num_;
        cyl_image_height_ =
            ((cyl_image_height_ - 1) / row_grid_num_ + 1) * row_grid_num_;

        // 初始化圆柱图像数组和额外的圆柱图像
        for (int i = 0; i < view_num_; ++i) {
            cyl_images_[i] =
                CylinderImageGPU(cyl_image_height_, cyl_image_width_);
        }
        extra_cyl_image_ =
            CylinderImageGPU4Channels(cyl_image_height_, cyl_image_width_);

        // 初始化图像对齐对象
        image_alignment_left_ = new ImageAlignmentCUDA(
            cyl_image_height_, cyl_image_width_, row_grid_num_, col_grid_num_);
        image_alignment_right_ = new ImageAlignmentCUDA(
            cyl_image_height_, cyl_image_width_, row_grid_num_, col_grid_num_);

        // 分配内存给新视图图像
        int render_h = cyl_image_height_,
            render_w = cyl_image_width_ / novel_images_num_;
        for (int i = 0; i < novel_images_num_; ++i) {
            uchar3* temp_image;
            cudaMalloc((void**)&temp_image,
                       sizeof(uchar3) * render_h * render_w);
            novel_images_.emplace_back(temp_image);
        }
    }

    // 清零相关图像和掩码内存
    checkCudaErrors(cudaMemset(extra_cyl_image_.image, 0,
                               extra_cyl_image_.width *
                                   extra_cyl_image_.height * sizeof(uchar4)));
    checkCudaErrors(cudaMemset(cyl_images_[1].image, 0,
                               cyl_images_[1].width * cyl_images_[1].height *
                                   sizeof(uchar3)));
    checkCudaErrors(cudaMemset(cyl_images_[1].mask, 0,
                               cyl_images_[1].width * cyl_images_[1].height *
                                   sizeof(uchar)));
    checkCudaErrors(cudaMemset(cyl_images_[0].image, 0,
                               cyl_images_[0].width * cyl_images_[0].height *
                                   sizeof(uchar3)));
    checkCudaErrors(cudaMemset(cyl_images_[0].mask, 0,
                               cyl_images_[0].width * cyl_images_[0].height *
                                   sizeof(uchar)));
    checkCudaErrors(cudaMemset(cyl_images_[2].image, 0,
                               cyl_images_[2].width * cyl_images_[2].height *
                                   sizeof(uchar3)));
    checkCudaErrors(cudaMemset(cyl_images_[2].mask, 0,
                               cyl_images_[2].width * cyl_images_[2].height *
                                   sizeof(uchar)));

    // 执行视图到圆柱图像的投影
    projToCylinderImage_cuda(views_, cyl_images_, *cyl_, cyl_image_width_,
                             cyl_image_height_);

#ifdef OUTPUT_CYL_IMAGE
    // 如果定义了输出圆柱图像，将结果保存到文件
    cv::Mat rgb_0, mask_0;
    static int cyl_image_count = 0;
    for (int i = 0; i < 3; i++) {
        cyl_images_[i].toCPU(rgb_0, mask_0);
        cv::imwrite("./output/out_image/cyl_image_" + std::to_string(i) + "_" +
                        std::to_string(cyl_image_count) + ".png",
                    rgb_0);
    }
    cyl_image_count++;
#endif
}

/**
 * 在CylinderStitcherGPU类中，执行图像对齐、寻找接缝和多带融合的步骤来缝合图像。
 * 此函数主要用于处理GPU上的图像缝合流程，包含对齐图像、查找接缝和应用融合。
 *
 * @param time 用于指定要处理的时间戳或帧数。
 */
void CylinderStitcherGPU::stitch_align_seam_blend(int time) {
    // 对图像进行对齐
    alignImages(time);
    // 确保所有对齐操作完成
    checkCudaErrors(cudaDeviceSynchronize());
    // 检查对齐过程中是否有错误发生
    checkCudaErrors(cudaGetLastError());

    // 查找图像接缝
    findSeam();
    // 确保接缝查找完成
    checkCudaErrors(cudaDeviceSynchronize());
    // 检查接缝查找过程中是否有错误
    checkCudaErrors(cudaGetLastError());

    // 应用多带融合来缝合图像
    MultiBandBlend_cuda(cyl_images_, seam_masks_);
    // 确保融合操作完成
    checkCudaErrors(cudaDeviceSynchronize());
    // 检查融合过程中是否有错误
    checkCudaErrors(cudaGetLastError());
}

/**
 * 将额外视图投影到圆柱图像并渲染到屏幕上
 *
 * @param time 当前时间戳，用于可能的时间依赖处理，但本函数体未直接使用此参数
 */
void CylinderStitcherGPU::stitch_project_to_screen(int time) {
    // 将额外视图转换为圆柱图像
    proj4ChannelsExtraViewToCylinderImage_cuda(extra_view_, extra_cyl_image_,
                                               *cyl_, cyl_image_width_,
                                               cyl_image_height_);
    // 将额外的圆柱图像与之前的圆柱图像融合
    BlendExtraViewToScreen4Channels_cuda(
        cyl_images_[1].image, extra_cyl_image_.image, cyl_images_[1].width,
        cyl_images_[1].height, 1.0);
    // 将融合后的圆柱图像渲染到屏幕上
    RenderToScreen_cuda(novel_view_intrins_, novel_view_extrin_Rs_,
                        cyl_images_[1], novel_images_, *cyl_, novel_view_pos_,
                        novel_images_num_);
    // 确保所有CUDA操作完成，检查错误
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

void CylinderStitcherGPU::getCylinderImageGPU(uchar3*& image, uchar*& mask,
                                              int& width, int& height) {
    image  = cyl_images_[1].image;
    mask   = cyl_images_[1].mask;
    width  = cyl_images_[1].width;
    height = cyl_images_[1].height;
}

void CylinderStitcherGPU::getCylinderImageCPU(cv::Mat& image, cv::Mat& mask) {
    uchar3* d_image;
    uchar* d_mask;
    int width, height;
    getCylinderImageGPU(d_image, d_mask, width, height);

    image = cv::Mat::zeros(height, width, CV_8UC3);
    mask  = cv::Mat::zeros(height, width, CV_8UC1);

    checkCudaErrors(cudaMemcpy(image.data, d_image,
                               width * height * sizeof(uchar3),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(mask.data, d_mask,
                               width * height * sizeof(uchar),
                               cudaMemcpyDeviceToHost));
}

/**
 * 该函数用于在CPU上获取最终图像和掩码。
 * 它通过将多个分块图像从GPU内存复制到CPU内存，并将它们拼接成一个最终图像。
 *
 * @param image 引用，用于存储最终的图像。
 * @param mask 引用，用于存储与最终图像对应的掩码。
 */
void CylinderStitcherGPU::getFinalImageCPU(cv::Mat& image, cv::Mat& mask) {
    // 计算每个新图像块的宽度和总高度
    int width  = cyl_images_[1].width / novel_images_num_;
    int height = cyl_images_[1].height;

    // 初始化最终图像和掩码为全零
    image = cv::Mat::zeros(height, cyl_images_[1].width, CV_8UC3);
    mask  = cv::Mat::zeros(height, cyl_images_[1].width, CV_8UC1);

    // 遍历所有新图像块，将它们从GPU内存复制到CPU内存，并拼接到最终图像上
    for (int i = 0; i < novel_images_num_; ++i) {
        cv::Mat temp_image = cv::Mat::zeros(height, width, CV_8UC3);
        // 从GPU设备内存复制图像数据到CPU主机内存
        checkCudaErrors(cudaMemcpy(temp_image.data, novel_images_[i],
                                   width * height * sizeof(uchar3),
                                   cudaMemcpyDeviceToHost));
        // 将复制的图像块粘贴到最终图像的相应位置
        temp_image.copyTo(image(cv::Rect(i * width, 0, width, height)));
    }
}
