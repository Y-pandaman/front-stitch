#pragma once

//#define OUTPUT_CYL_IMAGE

#include "cuda_utils.h"
#include "cylinder_stitcher.h"
#include "math_utils.h"
#include <opencv2/opencv.hpp>
#include <string>

struct PinholeCameraGPU {
    float fx, fy, cx, cy;
    float d0, d1, d2, d3;
    float* R = nullptr;
    float* T = nullptr;
    float* C = nullptr;

    PinholeCameraGPU() { }

    PinholeCameraGPU(float _fx, float _fy, float _cx, float _cy, float _d0,
                     float _d1, float _d2, float _d3, float* _R, float* _T,
                     float* _C) {
        fx = _fx;
        fy = _fy;
        cx = _cx;
        cy = _cy;
        d0 = _d0;
        d1 = _d1;
        d2 = _d2;
        d3 = _d3;
        R  = _R;
        T  = _T;
        C  = _C;
    }

    void printIntrin() {
        printf("fx = %f, fy = %f, cx = %f, cy = %f\n", fx, fy, cx, cy);
        printf("d0 = %f, d1 = %f, d2 = %f, d3 = %f\n", d0, d1, d2, d3);
    }

    void free() {
        if (R != nullptr) {
            cudaFree(R);
            cudaFree(T);
            cudaFree(C);
        }
    }

    void showPara() {
        float *R_, *T_, *C_;
        cudaHostAlloc((void**)&R_, sizeof(float) * 9, cudaHostAllocDefault);
        cudaHostAlloc((void**)&T_, sizeof(float) * 3, cudaHostAllocDefault);
        cudaHostAlloc((void**)&C_, sizeof(float) * 3, cudaHostAllocDefault);

        cudaMemcpy(R_, R, sizeof(float) * 9, cudaMemcpyDeviceToHost);
        cudaMemcpy(T_, T, sizeof(float) * 3, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_, C, sizeof(float) * 3, cudaMemcpyDeviceToHost);

        printf("fx: %f  fy: %f Cx: %f Cy: %f\n", fx, fy, cx, cy);
        printf("Rotation:\n");
        for (int i = 0; i < 3; i++) {
            printf("%f %f %f\n", R_[3 * i], R_[3 * i + 1], R_[3 * i + 2]);
        }
        printf("Translation Vector:\n");
        printf("%f %f %f\n", T_[0], T_[1], T_[2]);

        printf("Camera Center:\n");
        printf("%f %f %f\n", C_[0], C_[1], C_[2]);

        cudaFreeHost(R_);
        cudaFreeHost(T_);
        cudaFreeHost(C_);
    }

    inline __device__ float3 rotateVector(float3 v) {
        return make_float3(R[0] * v.x + R[1] * v.y + R[2] * v.z,
                           R[3] * v.x + R[4] * v.y + R[5] * v.z,
                           R[6] * v.x + R[7] * v.y + R[8] * v.z);
    }

    inline __device__ float3 rotateVector_inv(float3 v) {
        return make_float3(R[0] * v.x + R[3] * v.y + R[6] * v.z,
                           R[1] * v.x + R[4] * v.y + R[7] * v.z,
                           R[2] * v.x + R[5] * v.y + R[8] * v.z);
    }

    inline __device__ float3 getRay(int u, int v) {
        float3 dir = make_float3((u - cx) / fx, (v - cy) / fy, 1.0f);
        dir        = this->rotateVector_inv(dir);
        return normalize(dir);
    }

    inline __device__ float3 getCenter() {
        return make_float3(C[0], C[1], C[2]);
    }

    /**
     * 将世界坐标系中的点转换为像素坐标系中的点。
     *
     * @param x 世界坐标系中的点，格式为 float3(x, y, z)。
     * @return 像素坐标系中的点，如果点在相机视锥体之外，则返回 (-1, -1)。
     */
    inline __device__ float2 projWorldToPixel(float3 x) {
        // 将点 x 从世界坐标系转换到相机坐标系
        x.x -= C[0];
        x.y -= C[1];
        x.z -= C[2];
        // 对相机坐标系中的点进行旋转
        float3 cam = this->rotateVector(x);

        // 如果点在相机视锥体之外，则返回 (-1, -1)
        if (cam.z < 0)
            return make_float2(-1, -1);
        // 将相机坐标系中的点转换为像素坐标系中的点
        return make_float2(cam.x * fx / cam.z + cx, cam.y * fy / cam.z + cy);
    }

    /**
     * 将世界坐标系中的点转换为鱼眼镜头像素坐标系。
     *
     * @param x 世界坐标系中的点，格式为(float, float, float)。
     * @return 像素坐标系中的点，如果点在视野之外，则返回(-1, -1)。
     */
    inline __device__ float2 projWorldToPixelFishEye(float3 x) {
        // 将点从世界坐标系转换到相机坐标系
        x.x -= C[0];
        x.y -= C[1];
        x.z -= C[2];
        // 应用相机的旋转矩阵
        float3 cam = this->rotateVector(x);

        // 如果点在相机视锥体之外，则返回(-1, -1)
        if (cam.z < 0)
            return make_float2(-1, -1);

        // 计算归一化后的像素坐标
        float xx = cam.x / cam.z, yy = cam.y / cam.z;

        // 计算角度以及其各种幂次项
        float r     = sqrtf(xx * xx + yy * yy);
        float theta = atan(r);

        float theta2 = theta * theta, theta4 = theta2 * theta2,
              theta6 = theta4 * theta2, theta8 = theta4 * theta4;
        // 应用非线性校正模型，以模拟鱼眼镜头的效果
        float theta_d =
            theta * (1 + d0 * theta2 + d1 * theta4 + d2 * theta6 + d3 * theta8);

        // 计算最终的缩放因子，用于将角度映射到像素空间
        float scale = (r == 0) ? 1.0 : theta_d / r;
        // 计算并返回像素坐标
        return make_float2(fx * xx * scale + cx, fy * yy * scale + cy);
    }
};

struct ViewGPU {
    uchar3* image = nullptr;
    uchar* mask   = nullptr;
    int height, width;
    PinholeCameraGPU camera;
    ViewGPU() { }
    ViewGPU(uchar3* image_, uchar* mask_, int height_, int width_,
            PinholeCameraGPU camera_) {
        image  = image_;
        mask   = mask_;
        height = height_;
        width  = width_;
        camera = camera_;
    }

    bool toCPU(cv::Mat& image_, cv::Mat& mask_) {
        uchar3* data_rgb;
        uchar* data_mask;
        cudaHostAlloc((void**)&data_rgb, sizeof(uchar3) * height * width,
                      cudaHostAllocDefault);
        cudaHostAlloc((void**)&data_mask, sizeof(uchar) * height * width,
                      cudaHostAllocDefault);

        if (image != nullptr) {
            cudaMemcpy(data_rgb, image, sizeof(uchar3) * height * width,
                       cudaMemcpyDeviceToHost);
            image_ = cv::Mat(height, width, CV_8UC3, data_rgb);
        }
        if (mask != nullptr) {
            cudaMemcpy(data_mask, mask, sizeof(uchar) * height * width,
                       cudaMemcpyDeviceToHost);
            mask_ = cv::Mat(height, width, CV_8UC1, data_mask);
        }
        return true;
    }
};

struct ViewGPU4Channels {
    uchar4* image = nullptr;
    uchar* mask   = nullptr;
    int height, width;
    PinholeCameraGPU camera;
    ViewGPU4Channels() { }
    ViewGPU4Channels(uchar4* image_, uchar* mask_, int height_, int width_,
                     PinholeCameraGPU camera_) {
        image  = image_;
        mask   = mask_;
        height = height_;
        width  = width_;
        camera = camera_;
    }

    bool toCPU(cv::Mat& image_, cv::Mat& mask_) {
        uchar4* data_rgb;
        uchar* data_mask;
        cudaHostAlloc((void**)&data_rgb, sizeof(uchar4) * height * width,
                      cudaHostAllocDefault);
        cudaHostAlloc((void**)&data_mask, sizeof(uchar) * height * width,
                      cudaHostAllocDefault);

        if (image != nullptr) {
            cudaMemcpy(data_rgb, image, sizeof(uchar4) * height * width,
                       cudaMemcpyDeviceToHost);
            image_ = cv::Mat(height, width, CV_8UC4, data_rgb);
        }
        if (mask != nullptr) {
            cudaMemcpy(data_mask, mask, sizeof(uchar) * height * width,
                       cudaMemcpyDeviceToHost);
            mask_ = cv::Mat(height, width, CV_8UC1, data_mask);
        }
        return true;
    }
};

struct CylinderGPU {
    float* rotation;
    float* center;
    float r;
    float *global_theta, *global_phi;
    int2* boundary_pixel;
    int offset[4] = {0, 0, 0, 0};

    CylinderGPU() { }
    /**
     * CylinderGPU构造函数
     * 用于初始化一个GPU上的圆柱体对象，包括其旋转角度、中心点位置和半径。
     * 此构造函数还负责在GPU上分配内存，用于存储角度的最小和最大值。
     *
     * @param rotation_ 指向包含圆柱体旋转角度的浮点数组的指针。
     * @param center_ 指向包含圆柱体中心点坐标的浮点数组的指针。
     * @param r_ 圆柱体的半径。
     */

    CylinderGPU(float* rotation_, float* center_, float r_) {
        rotation = rotation_;
        center   = center_;
        r        = r_;
        // 在GPU上分配内存以存储全局角度变量
        cudaMalloc((void**)&global_theta, sizeof(float) * 2);
        cudaMalloc((void**)&global_phi, sizeof(float) * 2);

        // 定义并初始化角度的最小和最大值
        float min_theta = -3.141592653 / 180.0 * 80 + 0.001,
              max_theta = 3.141592653 / 180.0 * 80 - 0.001,
              min_phi   = -3.141592653 / 180.0 * 80,
              max_phi   = 3.141592653 / 180.0 * 80;
        // 将角度的最小和最大值复制到GPU上的全局变量
        cudaMemcpy(global_theta, &min_theta, sizeof(float) * 1,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(global_theta + 1, &max_theta, sizeof(float) * 1,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(global_phi, &min_phi, sizeof(float) * 1,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(global_phi + 1, &max_phi, sizeof(float) * 1,
                   cudaMemcpyHostToDevice);
        // 确保所有CUDA操作都已完成，并检查是否有错误发生
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
    }

    /**
     * 获取当前对象的中心点坐标。
     *
     * 本函数为内联函数，旨在减少函数调用开销，适用于需要频繁调用的场景。
     * 它不接受任何参数，直接返回一个float3类型的向量，包含中心点的x、y、z坐标。
     *
     * @return float3 返回一个包含中心点坐标的float3向量。
     */
    inline __device__ float3 getCenter() {
        // 直接构造并返回中心点的float3向量
        return make_float3(center[0], center[1], center[2]);
    }

    /**
     * @brief 对给定的向量进行旋转。
     *
     * 该函数使用一个预定义的3x3旋转矩阵来旋转输入的3D向量。
     *
     * @param v 待旋转的向量，其中包含x、y、z三个分量。
     * @return float3 旋转后的向量。
     */
    inline __device__ float3 rotateVector(float3 v) {
        // 应用旋转矩阵到输入向量
        return make_float3(rotation[0] * v.x + rotation[1] * v.y +
                               rotation[2] * v.z,   // 计算旋转后的x分量
                           rotation[3] * v.x + rotation[4] * v.y +
                               rotation[5] * v.z,   // 计算旋转后的y分量
                           rotation[6] * v.x + rotation[7] * v.y +
                               rotation[8] * v.z);   // 计算旋转后的z分量
    }

    /**
     * @brief 对给定的向量进行逆向旋转。
     *
     * 该函数通过一个预定义的旋转矩阵（全局变量rotation）来实现对输入向量的逆向旋转操作。
     * 注意：该函数设计为在CUDA设备端运行，适用于GPU加速计算。
     *
     * @param v 输入的待旋转向量，是一个3维向量（float3类型）。
     * @return float3 返回旋转后的向量。
     */
    inline __device__ float3 rotateVector_inv(float3 v) {
        // 应用旋转矩阵进行逆向旋转计算
        return make_float3(rotation[0] * v.x + rotation[3] * v.y +
                               rotation[6] * v.z,   // 计算旋转后的x分量
                           rotation[1] * v.x + rotation[4] * v.y +
                               rotation[7] * v.z,   // 计算旋转后的y分量
                           rotation[2] * v.x + rotation[5] * v.y +
                               rotation[8] * v.z);   // 计算旋转后的z分量
    }

    /**
     * 设置边界位置
     * @param boundary_loc
     * 包含边界位置的二维向量。每个子向量表示边界上一个特定部分的位置点集合。
     *                      子向量的顺序对应于offset的更新顺序。
     */
    void setBoundary(std::vector<std::vector<int2>> boundary_loc) {
        // 根据边界位置的子向量大小，计算每个部分的偏移量
        offset[1] = boundary_loc[0].size();
        offset[2] = boundary_loc[0].size() + boundary_loc[1].size();
        offset[3] = boundary_loc[0].size() + boundary_loc[1].size() +
                    boundary_loc[2].size();

        // 在GPU上分配内存，用于存储所有边界位置
        cudaMalloc((void**)&boundary_pixel, sizeof(int2) * offset[3]);

        // 将每个部分的边界位置从主机内存拷贝到GPU内存
        cudaMemcpy(boundary_pixel + offset[0], boundary_loc[0].data(),
                   sizeof(int2) * boundary_loc[0].size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(boundary_pixel + offset[1], boundary_loc[1].data(),
                   sizeof(int2) * boundary_loc[1].size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(boundary_pixel + offset[2], boundary_loc[2].data(),
                   sizeof(int2) * boundary_loc[2].size(),
                   cudaMemcpyHostToDevice);
    }
};

struct CylinderImageGPU4Channels {
    uchar4* image = nullptr;
    uchar* mask   = nullptr;
    int* uv;
    int height, width;
    CylinderImageGPU4Channels() { }

    CylinderImageGPU4Channels(uchar4* image_, uchar* mask_, int* uv_,
                              int height_, int width_) {
        image  = image_;
        mask   = mask_;
        uv     = uv_;
        height = height_;
        width  = width_;
    }

    /**
     * 构造函数：CylinderImageGPU4Channels
     * 用于分配GPU内存以存储四通道圆柱图像及其掩码和UV坐标。
     *
     * @param height_ 图像的高度。
     * @param width_ 图像的宽度。
     *
     * 注意：此构造函数不返回任何值，但会为类成员变量分配CUDA内存。
     */
    CylinderImageGPU4Channels(int height_, int width_) {
        height = height_;   // 设置图像高度
        width  = width_;    // 设置图像宽度

        // 分配GPU内存以存储四通道图像数据
        cudaMalloc((void**)&image, sizeof(uchar4) * height * width);
        // 分配GPU内存以存储图像的掩码数据
        cudaMalloc((void**)&mask, sizeof(uchar) * height * width);
        // 分配GPU内存以存储UV坐标
        cudaMalloc((void**)&uv, sizeof(int) * 2);
    }

    void reAllocateMem(int height_, int width_) {
        height = height_;
        width  = width_;
        cudaFree(image);
        cudaFree(mask);
        cudaMalloc((void**)&image, sizeof(uchar4) * height * width);
        cudaMalloc((void**)&mask, sizeof(uchar) * height * width);
    }

    bool toCPU(cv::Mat& image_, cv::Mat& mask_) {
        uchar4* data_rgb;
        uchar* data_mask;
        // int* data_uv;
        cudaHostAlloc((void**)&data_rgb, sizeof(uchar4) * height * width,
                      cudaHostAllocDefault);
        cudaHostAlloc((void**)&data_mask, sizeof(uchar) * height * width,
                      cudaHostAllocDefault);

        if (image != nullptr) {
            cudaMemcpy(data_rgb, image, sizeof(uchar4) * height * width,
                       cudaMemcpyDeviceToHost);
            image_ = cv::Mat(height, width, CV_8UC4, data_rgb);
        }
        if (mask != nullptr) {
            cudaMemcpy(data_mask, mask, sizeof(uchar) * height * width,
                       cudaMemcpyDeviceToHost);
            mask_ = cv::Mat(height, width, CV_8UC1, data_mask);
        }

        return true;
    }

    inline __device__ uchar getMaskValue(int x, int y) {
        if (x < 0 || y < 0 || x >= width || y >= height)
            return 0;
        return mask[y * width + x];
    }
    inline __device__ uchar4 getImageValue(int x, int y) {
        if (x < 0 || y < 0 || x >= width || y >= height)
            return make_uchar4(0, 0, 0, 0);
        return image[y * width + x];
    }
};

struct CylinderImageGPU {
    uchar3* image = nullptr;
    uchar* mask   = nullptr;
    int* uv;
    int height, width;
    CylinderImageGPU() { }

    CylinderImageGPU(uchar3* image_, uchar* mask_, int* uv_, int height_,
                     int width_) {
        image  = image_;
        mask   = mask_;
        uv     = uv_;
        height = height_;
        width  = width_;
    }

    /**
     * CylinderImageGPU构造函数
     * 用于初始化一个表示圆柱图像的GPU对象，包括分配图像、掩码和UV坐标所需的CUDA内存。
     *
     * @param height_ 图像的高度。
     * @param width_ 图像的宽度。
     */
    CylinderImageGPU(int height_, int width_) {
        height = height_;   // 设置图像高度
        width  = width_;    // 设置图像宽度

        // 分配用于存储图像数据的CUDA内存
        cudaMalloc((void**)&image, sizeof(uchar3) * height * width);
        // 分配用于存储掩码数据的CUDA内存
        cudaMalloc((void**)&mask, sizeof(uchar) * height * width);
        // 分配用于存储UV坐标的CUDA内存
        cudaMalloc((void**)&uv, sizeof(int) * 2);
    }

    void reAllocateMem(int height_, int width_) {
        height = height_;
        width  = width_;
        cudaFree(image);
        cudaFree(mask);
        cudaMalloc((void**)&image, sizeof(uchar3) * height * width);
        cudaMalloc((void**)&mask, sizeof(uchar) * height * width);
    }

    bool toCPU(cv::Mat& image_, cv::Mat& mask_) {
        uchar3* data_rgb;
        uchar* data_mask;
        // int* data_uv;
        cudaHostAlloc((void**)&data_rgb, sizeof(uchar3) * height * width,
                      cudaHostAllocDefault);
        cudaHostAlloc((void**)&data_mask, sizeof(uchar) * height * width,
                      cudaHostAllocDefault);

        if (image != nullptr) {
            cudaMemcpy(data_rgb, image, sizeof(uchar3) * height * width,
                       cudaMemcpyDeviceToHost);
            image_ = cv::Mat(height, width, CV_8UC3, data_rgb);
        }
        if (mask != nullptr) {
            cudaMemcpy(data_mask, mask, sizeof(uchar) * height * width,
                       cudaMemcpyDeviceToHost);
            mask_ = cv::Mat(height, width, CV_8UC1, data_mask);
        }

        return true;
    }

    /**
     * @brief 获取指定位置的掩码值
     *
     * 该函数用于从掩码数组中根据给定的坐标获取对应的掩码值。如果坐标超出掩码数组的范围，则返回0。
     *
     * @param x 指定位置的x坐标
     * @param y 指定位置的y坐标
     * @return uchar 返回指定位置的掩码值。如果坐标超出掩码数组的范围，则返回0。
     */
    inline __device__ uchar getMaskValue(int x, int y) {
        // 检查坐标是否在掩码数组的有效范围内
        if (x < 0 || y < 0 || x >= width || y >= height)
            return 0;
        // 返回指定位置的掩码值
        return mask[y * width + x];
    }

    /**
     * @brief 获取图像中指定坐标的像素值
     *
     * 该函数用于从图像中根据给定的x和y坐标获取对应的像素值。如果坐标超出图像范围，
     * 则返回一个全为0的像素值（黑像素）。
     *
     * @param x 图像中的x坐标
     * @param y 图像中的y坐标
     * @return uchar3
     * 像素值，以uchar3类型返回，分别代表红、绿、蓝三个颜色通道的值
     */
    inline __device__ uchar3 getImageValue(int x, int y) {
        // 检查坐标是否在图像范围内，若不在则返回黑色像素
        if (x < 0 || y < 0 || x >= width || y >= height)
            return make_uchar3(0, 0, 0);
        return image[y * width + x];
    }
};

struct SeamImageGPU {
    short3* image;
    short* mask;
    int height, width;
    SeamImageGPU() { }
    SeamImageGPU(short3* image_, short* mask_, int height_, int width_) {
        image  = image_;
        mask   = mask_;
        height = height_;
        width  = width_;
    }
    void free() {
        cudaFree(image);
        cudaFree(mask);
    }
};

class ImageAlignmentCUDA;

class CylinderStitcherGPU {
public:
    CylinderStitcherGPU(int view_num, int buffer_size);
    void setImages(uchar3* images, int width, int height, int view_num);
    void setExtraImageCuda(float* image_cuda, int width, int height);
    void setBoundingBoxesImage(uchar3* image, int width, int height);
    void setMasks(uchar* images, int width, int height, int view_num);
    void setCameras(std::vector<float4> intrins, std::vector<float4> distorts,
                    std::vector<float> extrins);
    void setExtraViewCamera(float4 intrin, std::vector<float> extrin_float);
    void alignImages(int time);
    void findSeam();
    void drawBoundingBoxes(cv::Mat& image, std::vector<float2>& boxes);
    void drawBoundingBoxes_2(cv::Mat& image, std::vector<float2>& boxes);
    void stitch_project_to_cyn(int time);
    void stitch_align_seam_blend(int time);
    void stitch_project_to_screen(int time);
    void getCylinderImageGPU(uchar3*& image, uchar*& mask, int& width,
                             int& height);
    void getCylinderImageCPU(cv::Mat& image, cv::Mat& mask);
    void getFinalImageCPU(cv::Mat& image, cv::Mat& mask);

public:
    float3 novel_view_pos_;
    int novel_images_num_ = 5;

    std::vector<ViewGPU> views_;
    ViewGPU4Channels extra_view_;
    uchar3* bounding_boxes_image_;
    std::vector<CylinderImageGPU> cyl_images_;
    CylinderImageGPU4Channels extra_cyl_image_;
    CylinderGPU* cyl_  = nullptr;
    uchar *d_buffer_1_ = nullptr, *d_buffer_2_ = nullptr,
          *d_buffer_3_ = nullptr;

    uchar* d_buffer_4channels_ = nullptr;
    int view_num_;

    ImageAlignmentCUDA *image_alignment_left_, *image_alignment_right_;
    int row_grid_num_ = 32, col_grid_num_ = 32;

    std::vector<uchar*> seam_masks_;
    std::vector<float4*> novel_view_intrins_;
    std::vector<float3*> novel_view_extrin_Rs_;

    Cylinder h_cyl_;
    std::vector<PinholeCamera> cameras_;
    std::vector<float4> h_novel_view_intrins_;
    std::vector<float3> h_novel_view_extrin_Rs_;

    std::vector<std::vector<int>> separate_lines_;

    int cyl_image_width_, cyl_image_height_;
    std::vector<uchar3*> novel_images_;
};