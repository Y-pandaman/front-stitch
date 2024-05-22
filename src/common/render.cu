#include "common/render.cuh"
#include "util/cuda_utils.h"

/**
 * 计算射线与圆柱体的交点。
 *
 * @param origin 射线的原点，类型为float3。
 * @param dir 射线的方向，类型为float3。
 * @param r 圆柱体的半径，类型为float。
 * @param intersection
 * 交点的坐标，类型为float3的引用，函数执行成功时会在这里存储交点坐标。
 * @return 如果射线与圆柱体相交，返回true；否则返回false。
 */
static inline __device__ bool RayCylinderIntersection_v2(float3 origin,
                                                         float3 dir, float r,
                                                         float3& intersection) {
    float3 abs_dir = fabs(dir);   // 计算射线方向的绝对值，用于后续判断。

    // 处理垂直于x轴的情况。
    if (abs_dir.x < CLOSE_ZERO) {
        intersection.x = origin.x;   // x坐标不变。
        intersection.z =
            sqrtf(powf(r, 2.0f) -
                  powf(intersection.x, 2.0f));   // 根据半径和x坐标计算z坐标。

        // 根据射线方向调整z坐标的符号。
        if ((intersection.z - origin.z) / dir.z < 0)
            intersection.z *= -1.0f;

        // 处理垂直于z轴的情况。
    } else if (abs_dir.z < CLOSE_ZERO) {
        intersection.z = origin.z;   // z坐标不变。
        intersection.x =
            sqrtf(powf(r, 2.0f) -
                  powf(intersection.z, 2.0f));   // 根据半径和z坐标计算x坐标。

        // 根据射线方向调整x坐标的符号。
        if ((intersection.x - origin.x) / dir.x < 0)
            intersection.x *= -1.0f;

        // 处理一般情况。
    } else {
        float k = dir.z / dir.x;             // 计算斜率。
        float b = origin.z - k * origin.x;   // 计算截距。

        // 判断射线与圆柱体是否有交点。
        float check =
            powf(2.0f * k * b, 2.0f) - 4 * (1 + k * k) * (b * b - r * r);
        if (check < 0)
            return false;

        float item = sqrtf(check);   // 计算平方根。

        // 计算两个交点的x和z坐标。
        float x1 = (-2.0f * k * b + item) / (2 * (1 + k * k));
        float x2 = (-2.0f * k * b - item) / (2 * (1 + k * k));
        float z1 = k * x1 + b;
        float z2 = k * x2 + b;

        // 计算tanθ，θ为射线与水平面的夹角。
        float tan_theta = dir.y / sqrtf(dir.x * dir.x + dir.z * dir.z);

        // 判断两个交点是否在射线的前进方向上。
        bool cross_1 =
            (x1 - origin.x) / dir.x > 0 && (z1 - origin.z) / dir.z > 0;
        bool cross_2 =
            (x2 - origin.x) / dir.x > 0 && (z2 - origin.z) / dir.z > 0;

        // 如果两个交点都在前进方向上，选择距离原点更远的那个交点。
        if (cross_1 && cross_2) {
            float t1_xz =
                sqrtf(powf(x1 - origin.x, 2.0f) + powf(z1 - origin.z, 2.0f));
            float t2_xz =
                sqrtf(powf(x2 - origin.x, 2.0f) + powf(z2 - origin.z, 2.0f));
            if (t1_xz > t2_xz) {
                intersection.x = x1;
                intersection.z = z1;
                intersection.y = origin.y + t1_xz * tan_theta;
            } else {
                intersection.x = x2;
                intersection.z = z2;
                intersection.y = origin.y + t2_xz * tan_theta;
            }

            // 如果只有一个交点在前进方向上，选择该交点。
        } else {
            if (cross_1) {
                intersection.x = x1;
                intersection.z = z1;
            } else {
                intersection.x = x2;
                intersection.z = z2;
            }
            // 计算交点到原点的距离，并据此计算y坐标。
            float t_xz     = sqrtf(powf(intersection.x - origin.x, 2.0f) +
                               powf(intersection.z - origin.z, 2.0f));
            intersection.y = origin.y + t_xz * tan_theta;
        }
    }
    return true;   // 表示射线与圆柱体相交。
}

/**
 * 一维插值函数
 * 通过线性插值，在两个给定的uchar3类型向量之间，根据参数x的比例来计算插值结果。
 *
 * @param v1 第一个输入向量
 * @param v2 第二个输入向量
 * @param x 插值比例，范围为[0.0, 1.0]，其中0.0表示完全使用v1，1.0表示完全使用v2
 * @return 返回一个uchar3类型的插值向量
 */
static inline __device__ __host__ uchar3 interpolate1D(uchar3 v1, uchar3 v2,
                                                       float x) {
    return make_uchar3((uchar)((float)v1.x * (1.0f - x) + (float)v2.x * x),
                       (uchar)((float)v1.y * (1.0f - x) + (float)v2.y * x),
                       (uchar)((float)v1.z * (1.0f - x) + (float)v2.z * x));
}

/**
 * 二维插值函数
 * 通过两次一维插值操作，在四个给定的uchar3类型向量构成的二维格子中，根据参数x和y的比例来计算插值结果。
 *
 * @param v1 第一个输入向量，对应二维格子的左上角
 * @param v2 第二个输入向量，对应二维格子的右上角
 * @param v3 第三个输入向量，对应二维格子的左下角
 * @param v4 第四个输入向量，对应二维格子的右下角
 * @param x 插值比例，在x轴上的比例，范围为[0.0, 1.0]
 * @param y 插值比例，在y轴上的比例，范围为[0.0, 1.0]
 * @return 返回一个uchar3类型的插值向量
 */
static inline __device__ __host__ uchar3 interpolate2D(uchar3 v1, uchar3 v2,
                                                       uchar3 v3, uchar3 v4,
                                                       float x, float y) {
    // 第一次插值，在x轴方向上
    uchar3 s = interpolate1D(v1, v2, x);
    // 第二次插值，在y轴方向上，输入为前一次插值的结果
    uchar3 t = interpolate1D(v3, v4, x);
    // 第二次插值的结果作为输入，在xy平面上进行最后的插值
    return interpolate1D(s, t, y);
}

/**
 * 在给定的图像中，根据指定的像素坐标，使用双线性插值计算并返回一个像素的颜色。
 *
 * @param src 指向图像数据的指针，图像数据以 uchar3 格式存储（RGB 或 BGR）。
 * @param h 图像的高度。
 * @param w 图像的宽度。
 * @param pixel 指定的像素坐标，为一个 float2 类型，x 和 y
 * 分别表示像素点的横纵坐标。
 * @return 计算得到的像素颜色，以 uchar3 格式返回。
 */
static inline __device__ uchar3 Bilinear(uchar3* src, int h, int w,
                                         float2 pixel) {
    int x0  = (int)pixel.x;
    int y0  = (int)pixel.y;
    int x1  = x0 + 1;
    int y1  = y0 + 1;
    float x = pixel.x - x0;
    float y = pixel.y - y0;

    // 计算四个邻近像素的索引
    int idx00 = x0 + y0 * w;
    int idx01 = idx00 + w;
    int idx10 = idx00 + 1;
    int idx11 = idx01 + 1;

    // 条件边界检查 TODO: 需要实现边界条件的检查，以确保索引不会超出图像边界

    // 执行双线性插值计算
    return interpolate2D(src[idx00], src[idx10], src[idx01], src[idx11], x, y);
}

/**
 * 在CUDA设备上执行的一个内核函数，用于将一个四通道图像与另一个图像以特定权重进行融合。
 *
 * @param dst_cyl_img
 * 指向目标图像（圆柱坐标图像）数据的指针，类型为uchar3，融合结果将保存在此图像中。
 * @param src_cyl_img
 * 指向源图像（圆柱坐标图像）数据的指针，类型为uchar4，此图像的数据将与目标图像融合。
 * @param width 图像的宽度。
 * @param height 图像的高度。
 * @param w 源图像的权重，取值范围为[0,1]，用于控制源图像与目标图像的融合程度。
 */
__global__ void blend_extra_view_4channels_kernel(uchar3* dst_cyl_img,
                                                  uchar4* src_cyl_img,
                                                  int width, int height,
                                                  float w) {
    // 计算当前线程处理的像素索引
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    // 仅处理有效像素
    if (pixelIdx < width * height) {
        // 计算源图像的权重和目标图像的权重
        float w_1 = (float)src_cyl_img[pixelIdx].w / 255.0;
        float w_2 = 1.0 - w_1;
        // 根据权重融合像素的R、G、B值
        dst_cyl_img[pixelIdx].x = (w_1 * (float)src_cyl_img[pixelIdx].z +
                                   w_2 * (float)dst_cyl_img[pixelIdx].x);
        dst_cyl_img[pixelIdx].y = (w_1 * (float)src_cyl_img[pixelIdx].y +
                                   w_2 * (float)dst_cyl_img[pixelIdx].y);
        dst_cyl_img[pixelIdx].z = (w_1 * (float)src_cyl_img[pixelIdx].x +
                                   w_2 * (float)dst_cyl_img[pixelIdx].z);
    }
}

// __global__ void blend_extra_view_kernel(uchar3* dst_cyl_img,
//                                         uchar3* src_cyl_img, int width,
//                                         int height, float w) {
//     int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (pixelIdx < width * height) {
//         if (src_cyl_img[pixelIdx].x > 0 || src_cyl_img[pixelIdx].y > 0 ||
//             src_cyl_img[pixelIdx].z > 0) {
//             float w_1 =
//                 max(src_cyl_img[pixelIdx].x,
//                     max(src_cyl_img[pixelIdx].y, src_cyl_img[pixelIdx].z)) /
//                 255.0 * w;
//             float w_2               = 1.0 - w_1;
//             dst_cyl_img[pixelIdx].x = (w_1 * (float)src_cyl_img[pixelIdx].z +
//                                        w_2 * (float)dst_cyl_img[pixelIdx].x);
//             dst_cyl_img[pixelIdx].y = (w_1 * (float)src_cyl_img[pixelIdx].y +
//                                        w_2 * (float)dst_cyl_img[pixelIdx].y);
//             dst_cyl_img[pixelIdx].z = (w_1 * (float)src_cyl_img[pixelIdx].x +
//                                        w_2 * (float)dst_cyl_img[pixelIdx].z);
//         }
//     }
// }

/**
 * @brief 核心渲染函数，用于计算每个像素点的颜色。
 *
 * @param intrin 存储相机内参的浮点数数组，数组中包含fx, fy, cx, cy。
 * @param center_view_intrin 中心视图的相机内参浮点数数组。
 * @param extrin_R 存储相机外参旋转矩阵的浮点数数组。
 * @param cyl_img 缩放后的圆柱图像。
 * @param src_h 源图像的高度。
 * @param src_w 源图像的宽度。
 * @param cyl 圆柱体的参数。
 * @param ox 相机中心的x坐标。
 * @param oy 相机中心的y坐标。
 * @param oz 相机中心的z坐标。
 * @param dst 目标图像，存储渲染后的颜色。
 * @param idx 当前视图的索引。
 * @param height 目标图像的高度。
 * @param width 目标图像的宽度。
 * @param view_num 视图的数量。
 * @param is_center_view 是否为中心视图。
 */
__global__ void render_kernel(float4* intrin, float4* center_view_intrin,
                              float3* extrin_R, uchar3* cyl_img, int src_h,
                              int src_w, CylinderGPU cyl, float ox, float oy,
                              float oz, uchar3* dst, int idx,   // idx = 0, 1, 2
                              int height, int width, int view_num,
                              bool is_center_view) {
    // 计算线程索引和总线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 计算图像的theta和phi范围及步长
    float image_theta_range =
        (cyl.global_theta[1] - cyl.global_theta[0]) / view_num;
    float image_phi_range = cyl.global_phi[1] - cyl.global_phi[0];

    float theta_step = (cyl.global_theta[1] - cyl.global_theta[0]) / src_w;
    float phi_step   = image_phi_range / src_h;

    // 计算当前视图的中心点参数
    float center_theta = cyl.global_theta[0] + image_theta_range / 2.0f +
                         idx * image_theta_range;
    float center_phi = cyl.global_phi[0] + image_phi_range / 2.0f;
    float3 p_cyl = make_float3(sinf(center_theta) * cyl.r, center_phi * cyl.r,
                               cosf(center_theta) * cyl.r);
    float3 inter_loc = cyl.rotateVector_inv(p_cyl) + cyl.getCenter();

    // 计算相机中心和坐标轴
    float3 camera_center = make_float3(ox, oy, oz);
    float3 Y_axis =
        make_float3(cyl.rotation[3], cyl.rotation[4], cyl.rotation[5]);
    float3 Z_axis = normalize(inter_loc - camera_center);
    float3 X_axis = normalize(cross(Y_axis, Z_axis));
    Z_axis        = normalize(cross(X_axis, Y_axis));
    extrin_R[0]   = make_float3(X_axis.x, Y_axis.x, Z_axis.x);
    extrin_R[1]   = make_float3(X_axis.y, Y_axis.y, Z_axis.y);
    extrin_R[2]   = make_float3(X_axis.z, Y_axis.z, Z_axis.z);

    // 计算视锥的两个边界点
    float3 p1 =
        make_float3(sinf(center_theta - image_theta_range / 2.0f) * cyl.r,
                    cyl.global_phi[0] * cyl.r,
                    cosf(center_theta - image_theta_range / 2.0f) * cyl.r);
    float3 p2 =
        make_float3(sinf(center_theta + image_theta_range / 2.0f) * cyl.r,
                    cyl.global_phi[1] * cyl.r,
                    cosf(center_theta + image_theta_range / 2.0f) * cyl.r);
    p1 = cyl.rotateVector_inv(p1) + cyl.getCenter() - camera_center;
    p2 = cyl.rotateVector_inv(p2) + cyl.getCenter() - camera_center;
    p1 = rotateVector(X_axis, Y_axis, Z_axis, p1);
    p2 = rotateVector(X_axis, Y_axis, Z_axis, p2);

    // 计算相机内参
    float fx, fy, cx, cy;
    fx = (1.0f - width) / (p1.x / p1.z - p2.x / p2.z);
    cx = -1.0f * p1.x / p1.z * fx;
    fy = (1.0f - height) / (p1.y / p1.z - p2.y / p2.z);
    cy = -1.0f * p1.y / p1.z * fy;
    if (!is_center_view) {
        fy = center_view_intrin->y;
        cy = center_view_intrin->w;
    }
    intrin->x = fx;
    intrin->y = fy;
    intrin->z = cx;
    intrin->w = cy;

    // 遍历所有像素，计算每个像素的颜色
    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 计算像素点对应的光线方向
        float3 dir = make_float3((x - cx) / fx, (y - cy) / fy, 1.0f);

        // 将方向向量转换到圆柱坐标系下
        dir = rotateVector_inv(X_axis, Y_axis, Z_axis, dir);
        dir = cyl.rotateVector(normalize(dir));

        // 计算光线起点
        float3 origin = cyl.rotateVector(camera_center - cyl.getCenter());

        // 计算光线与圆柱的交点
        float3 P;
        bool success = RayCylinderIntersection_v2(origin, dir, cyl.r, P);

        // 如果光线与圆柱相交，计算像素颜色
        if (success) {
            float theta = clamp(atan2f(P.x, P.z), -3.141592653 / 2 + 0.001,
                                3.141592653 / 2 - 0.001);
            float phi   = P.y / cyl.r;

            float u = (theta - cyl.global_theta[0]) / theta_step;
            float v = (phi - cyl.global_phi[0]) / phi_step;

            // 使用双线性插值获取像素颜色
            if (u >= 0 && v >= 0 && u < src_w - 1 && v < src_h - 1) {
                uchar3 color =
                    Bilinear(cyl_img, src_h, src_w, make_float2(u, v));
                dst[pixelIdx] = color;
            } else {
                dst[pixelIdx] = make_uchar3(0, 0, 0);
            }
        } else {
            dst[pixelIdx] = make_uchar3(0, 0, 0);
        }

        pixelIdx += total_thread;
    }
}

/**
 * 在CUDA平台上，将额外视图的四个通道融合到屏幕图像上。
 *
 * @param dst_cyl_img
 * 指向目标圆柱图像（已融合图像）的uchar3指针。该图像格式为RGB。
 * @param src_cyl_img
 * 指向源圆柱图像（待融合图像）的uchar4指针。该图像格式为RGBA。
 * @param width 图像的宽度。
 * @param height 图像的高度。
 * @param w 融合权重，取值范围为0到1，表示源图像的融合程度。
 */
__host__ void BlendExtraViewToScreen4Channels_cuda(uchar3* dst_cyl_img,
                                                   uchar4* src_cyl_img,
                                                   int width, int height,
                                                   float w) {
    // 根据图像大小计算线程块和网格的数量
    int num_thread = 512;   // 线程块大小
    int num_block  = min(65535, (height * width + num_thread - 1) /
                                   num_thread);   // 网格大小

    // 调用CUDA内核函数进行图像融合
    blend_extra_view_4channels_kernel<<<num_block, num_thread>>>(
        dst_cyl_img, src_cyl_img, width, height, w);

    // 等待所有CUDA任务完成，并检查是否有错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

/**
 * 使用CUDA将图像渲染到屏幕。
 *
 * @param novel_view_intrins 存储了新视图内在参数（相机内参）的向量。
 * @param novel_view_extrin_Rs 存储了新视图外部参数（相机位姿）的向量。
 * @param cyl_img 表示圆柱体图像的GPU数据结构。
 * @param novel_images 存储了新视图图像数据的向量。
 * @param cylinder 表示圆柱体的GPU数据结构。
 * @param origin 圆柱体的原点坐标。
 * @param novel_images_num 新视图的数量。
 *
 * 函数首先检查并为novel_view_intrins和novel_view_extrin_Rs分配CUDA内存，
 * 然后调用CUDA内核函数render_kernel来渲染图像。
 * 最后，同步CUDA设备以确保所有操作都完成，并检查是否有CUDA错误发生。
 */
__host__ void RenderToScreen_cuda(std::vector<float4*>& novel_view_intrins,
                                  std::vector<float3*>& novel_view_extrin_Rs,
                                  CylinderImageGPU cyl_img,
                                  std::vector<uchar3*> novel_images,
                                  CylinderGPU cylinder, float3 origin,
                                  int novel_images_num) {
    // 如果没有提供新视图的内在参数，分配CUDA内存并初始化。
    if (novel_view_intrins.size() == 0) {
        for (int i = 0; i < novel_images_num; i++) {
            float4* temp_1;
            checkCudaErrors(cudaMalloc((void**)&temp_1, sizeof(float4)));
            novel_view_intrins.emplace_back(temp_1);
            float3* temp_2;
            checkCudaErrors(cudaMalloc((void**)&temp_2, sizeof(float3) * 3));
            novel_view_extrin_Rs.emplace_back(temp_2);
        }
    }

    // 计算CUDA线程块和块数以适配渲染任务。
    int num_thread = 512;
    int height = cyl_img.height, width = cyl_img.width / novel_images_num;
    int num_block = min(65535, (height * width + num_thread - 1) / num_thread);

    // 首先针对中间视图执行渲染。
    render_kernel<<<num_block, num_thread>>>(
        novel_view_intrins[novel_images_num / 2],
        novel_view_intrins[novel_images_num / 2],
        novel_view_extrin_Rs[novel_images_num / 2], cyl_img.image,
        cyl_img.height, cyl_img.width, cylinder, origin.x, origin.y, origin.z,
        novel_images[novel_images_num / 2], novel_images_num / 2, height, width,
        novel_images_num, true);

    // 对于剩余的视图，执行渲染。
    for (int i = 0; i < novel_images_num; i++) {
        if (i != (novel_images_num / 2))
            render_kernel<<<num_block, num_thread>>>(
                novel_view_intrins[i], novel_view_intrins[novel_images_num / 2],
                novel_view_extrin_Rs[i], cyl_img.image, cyl_img.height,
                cyl_img.width, cylinder, origin.x, origin.y, origin.z,
                novel_images[i], i, height, width, novel_images_num, false);
    }

    // 同步CUDA设备并检查错误。
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}