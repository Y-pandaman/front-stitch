#include "cuda_utils.h"
#include "project_to_cylinder.cuh"
#include <thrust/extrema.h>

/**
 * 一维插值函数
 * 该函数用于在两个给定的 uchar3 值之间进行线性插值，返回一个新的 uchar3 值。
 *
 * @param v1 第一个 uchar3 插值点
 * @param v2 第二个 uchar3 插值点
 * @param x 插值系数，x 在 0.0f 到 1.0f 之间，表示从 v1 向 v2 的插值比例
 * @return 一个新的 uchar3 值，它是根据插值系数 x 在 v1 和 v2 之间线性插值得到的
 */
static inline __device__ __host__ uchar3 interpolate1D(uchar3 v1, uchar3 v2,
                                                       float x) {
    // 对每个颜色通道（x, y, z）进行线性插值计算
    return make_uchar3((uchar)((float)v1.x * (1.0f - x) + (float)v2.x * x),
                       (uchar)((float)v1.y * (1.0f - x) + (float)v2.y * x),
                       (uchar)((float)v1.z * (1.0f - x) + (float)v2.z * x));
}

/**
 * 一维插值函数，用于在两个 uchar4 值之间进行线性插值。
 *
 * @param v1 第一个 uchar4 输入值，作为插值的起点。
 * @param v2 第二个 uchar4 输入值，作为插值的终点。
 * @param x 插值系数，取值范围为 [0.0, 1.0]，其中 0.0 表示完全使用 v1 的值，1.0
 * 表示完全使用 v2 的值。
 * @return 返回一个 uchar4 类型的插值结果，其中每个通道都进行了线性插值计算。
 */
static inline __device__ __host__ uchar4 interpolate1D4Channels(uchar4 v1,
                                                                uchar4 v2,
                                                                float x) {
    // 对每个颜色通道执行线性插值计算，并将结果转换为 uchar 类型
    return make_uchar4((uchar)((float)v1.x * (1.0f - x) + (float)v2.x * x),
                       (uchar)((float)v1.y * (1.0f - x) + (float)v2.y * x),
                       (uchar)((float)v1.z * (1.0f - x) + (float)v2.z * x),
                       (uchar)((float)v1.w * (1.0f - x) + (float)v2.w * x));
}

/**
 * 对四个二维采样点进行插值，返回插值后的颜色值。
 * 该函数首先在x方向对给定的四个采样点进行一次插值，然后在y方向对插值结果再次进行插值。
 *
 * @param v1 采样点1的颜色值。
 * @param v2 采样点2的颜色值。
 * @param v3 采样点3的颜色值。
 * @param v4 采样点4的颜色值。
 * @param x  在x方向上的插值系数。
 * @param y  在y方向上的插值系数。
 * @return 插值后的颜色值，以uchar4类型返回。
 */
static inline __device__ __host__ uchar4 interpolate2D4Channels(
    uchar4 v1, uchar4 v2, uchar4 v3, uchar4 v4, float x, float y) {
    // 在x方向上对v1和v2进行插值，得到中间颜色值s
    uchar4 s = interpolate1D4Channels(v1, v2, x);
    // 在x方向上对v3和v4进行插值，得到中间颜色值t
    uchar4 t = interpolate1D4Channels(v3, v4, x);
    // 在y方向上对中间颜色值s和t进行插值，得到最终的插值颜色值
    return interpolate1D4Channels(s, t, y);
}

/**
 * 二维插值函数
 * 通过给定的四个顶点颜色（v1, v2, v3, v4）和一个二维坐标（x,
 * y）来插值计算出该点的颜色。
 *
 * @param v1 顶点1的颜色，位于二维空间的一个角点
 * @param v2 顶点2的颜色，位于二维空间的一个角点
 * @param v3 顶点3的颜色，位于二维空间的一个角点
 * @param v4 顶点4的颜色，位于二维空间的一个角点
 * @param x  在二维空间中x轴的坐标比例，用于确定水平方向上的插值
 * @param y  在二维空间中y轴的坐标比例，用于确定垂直方向上的插值
 * @return 返回插值后得到的颜色，作为一个uchar3类型（包含红、绿、蓝三个分量）
 */
static inline __device__ __host__ uchar3 interpolate2D(uchar3 v1, uchar3 v2,
                                                       uchar3 v3, uchar3 v4,
                                                       float x, float y) {
    // 在x方向上进行插值，得到一个中间颜色s
    uchar3 s = interpolate1D(v1, v2, x);
    // 在x方向上再次进行插值，得到另一个中间颜色t
    uchar3 t = interpolate1D(v3, v4, x);
    // 在y方向上对中间颜色s和t进行插值，最终得到所需的颜色
    return interpolate1D(s, t, y);
}

/**
 * 在给定的图像中，通过双线性插值计算指定像素点的颜色。
 *
 * @param src 指向图像数据的指针，图像数据以uchar4（RGBA）格式存储。
 * @param h 图像的高度。
 * @param w 图像的宽度。
 * @param pixel 指定的像素点坐标，为浮点型，以便支持亚像素精度。
 * @return 计算得到的指定像素点的颜色，以uchar4（RGBA）格式返回。
 */
static inline __device__ uchar4 Bilinear4Channels(uchar4* src, int h, int w,
                                                  float2 pixel) {
    // 将像素坐标向下取整得到x0和y0
    int x0 = (int)pixel.x;
    int y0 = (int)pixel.y;
    // 计算相邻的像素点坐标x1和y1
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    // 计算像素点在x和y方向上的偏差
    float x = pixel.x - x0;
    float y = pixel.y - y0;

    // 根据坐标计算像素在图像数组中的索引
    int idx00 = x0 + y0 * w;
    int idx01 = idx00 + w;   // 同一行下一个像素的索引
    int idx10 = idx00 + 1;   // 下一行同一个列的像素索引
    int idx11 = idx01 + 1;   // 下一行下一个列的像素索引

    // 使用双线性插值计算指定像素的颜色
    return interpolate2D4Channels(src[idx00], src[idx10], src[idx01],
                                  src[idx11], x, y);
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

    // TODO: 需要实现边界条件的检查，以确保索引不会超出图像边界

    // 执行双线性插值计算
    return interpolate2D(src[idx00], src[idx10], src[idx01], src[idx11], x, y);
}

/**
 * 在CUDA设备上执行的BackProjToSrc4Channels_kernel函数。
 * 该函数用于将图像从圆柱坐标系重新投影回源图像坐标系。
 *
 * @param src_color 源图像的彩色数据，类型为uchar4。
 * @param mask 源图像的掩码，如果不需要则可以为nullptr。
 * @param src_h 源图像的高度。
 * @param src_w 源图像的宽度。
 * @param cyl 圆柱模型的参数。
 * @param cam 镜头模型的参数。
 * @param cyl_image 投影回圆柱坐标系后的图像数据。
 * @param cyl_mask 圆柱坐标系图像的掩码，如果不需要则可以为nullptr。
 * @param uv 用于存储第一个像素的UV坐标。
 * @param global_min_theta 全局最小的theta值。
 * @param global_min_phi 全局最小的phi值。
 * @param global_max_theta 全局最大的theta值。
 * @param global_max_phi 全局最大的phi值。
 * @param min_theta 当前块处理的最小theta值。
 * @param min_phi 当前块处理的最小phi值。
 * @param height 投影图像的高度。
 * @param width 投影图像的宽度。
 * @param is_fisheye 是否使用鱼眼镜头模型。
 *
 * 注：该函数不返回任何值，通过指针参数直接修改数据。
 */
static __global__ void BackProjToSrc4Channels_kernel(
    uchar4* src_color, uchar* mask, int src_h, int src_w, CylinderGPU cyl,
    PinholeCameraGPU cam, uchar4* cyl_image, uchar* cyl_mask, int* uv,
    float* global_min_theta, float* global_min_phi, float* global_max_theta,
    float* global_max_phi, float* min_theta, float* min_phi, int height,
    int width, bool is_fisheye) {
    // 计算线程索引和总线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 计算theta和phi的步长
    float step_x = ((*global_max_theta) - (*global_min_theta)) / width;
    float step_y = ((*global_max_phi) - (*global_min_phi)) / height;

    // 遍历所有像素
    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 计算当前像素对应的theta和phi值
        float theta = x * step_x + *min_theta;
        float phi   = y * step_y + *min_phi;

        // 根据theta和phi计算世界坐标
        float3 P =
            make_float3(cyl.r * sinf(theta), cyl.r * phi, cyl.r * cosf(theta));

        // 将世界坐标投影回像素坐标
        float2 pixel;
        if (is_fisheye) {
            pixel = cam.projWorldToPixelFishEye(cyl.rotateVector_inv(P) +
                                                cyl.getCenter());
        } else {
            pixel =
                cam.projWorldToPixel(cyl.rotateVector_inv(P) + cyl.getCenter());
        }

        // 初始化颜色和掩码值
        uchar4 color = make_uchar4(0, 0, 0, 0);
        uchar m      = 0;

        // 无掩码时处理像素
        if (mask == nullptr) {
            // 使用双线性插值获取源图像颜色
            if (pixel.x >= 0 && pixel.y >= 0 && pixel.x < src_w - 1 &&
                pixel.y < src_h - 1) {
                color = Bilinear4Channels(src_color, src_h, src_w, pixel);
            }

            // 计算在圆柱图像中的位置
            int row = (phi - (*global_min_phi)) / step_y + 0.5;
            int col = (theta - (*global_min_theta)) / step_x + 0.5;
            if (x == 0 && y == 0) {
                uv[0] = 0;
                uv[1] = 0;
            }

            // 更新圆柱图像数据
            if (row >= 0 && row < height && col >= 0 && col < width) {
                cyl_image[row * width + col] = color;
            }
        } else {   // 有掩码时处理像素
            // 使用双线性插值获取源图像颜色和掩码值
            if (pixel.x >= 0 && pixel.y >= 0 && pixel.x < src_w - 1 &&
                pixel.y < src_h - 1) {
                color = Bilinear4Channels(src_color, src_h, src_w, pixel);
                m = mask[(int)(pixel.y + 0.5f) * src_w + (int)(pixel.x + 0.5f)];
            }

            // 计算在圆柱图像中的位置
            int row = (phi - (*global_min_phi)) / step_y + 0.5;
            int col = (theta - (*global_min_theta)) / step_x + 0.5;
            if (x == 0 && y == 0) {
                uv[0] = 0;
                uv[1] = 0;
            }

            // 更新圆柱图像数据和掩码
            if (row >= 0 && row < height && col >= 0 && col < width) {
                cyl_image[row * width + col] = color;
                cyl_mask[row * width + col]  = m;
            }
        }

        // 移动到下一个像素
        pixelIdx += total_thread;
    }
}

/**
 * 在CUDA设备上执行的后台投影到源图像的核函数。
 *
 * @param src_color 源图像的彩色数据数组。
 * @param mask 源图像的掩码，如果为nullptr，则不使用掩码。
 * @param src_h 源图像的高度。
 * @param src_w 源图像的宽度。
 * @param cyl 表示圆柱体的对象。
 * @param cam 表示针孔相机的对象。
 * @param cyl_image 投影到圆柱体图像的结果数组。
 * @param cyl_mask
 * 圆柱体图像的掩码结果数组，如果源mask为nullptr，则此参数也相应不使用。
 * @param uv 存储在计算过程中使用的临时UV坐标。
 * @param global_min_theta 全局最小的theta角。
 * @param global_min_phi 全局最小的phi角。
 * @param global_max_theta 全局最大的theta角。
 * @param global_max_phi 全局最大的phi角。
 * @param min_theta 当前计算区域的最小theta角。
 * @param min_phi 当前计算区域的最小phi角。
 * @param height 投影图像的高度。
 * @param width 投影图像的宽度。
 * @param is_fisheye 表示是否使用鱼眼镜头模型。
 *
 * 该函数通过计算每个像素的世界坐标，并将其投影回源图像，来生成一个从特定视点看圆柱体表面的图像。
 * 根据是否提供mask，可以进行普通的或带有掩码的图像处理。
 */
static __global__ void
BackProjToSrc_kernel(uchar3* src_color, uchar* mask, int src_h, int src_w,
                     CylinderGPU cyl, PinholeCameraGPU cam, uchar3* cyl_image,
                     uchar* cyl_mask, int* uv, float* global_min_theta,
                     float* global_min_phi, float* global_max_theta,
                     float* global_max_phi, float* min_theta, float* min_phi,
                     int height, int width, bool is_fisheye) {
    // 计算当前线程处理的像素索引和总线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 计算theta和phi的步长
    float step_x = ((*global_max_theta) - (*global_min_theta)) / width;
    float step_y = ((*global_max_phi) - (*global_min_phi)) / height;

    // 遍历所有像素
    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 计算当前像素对应的theta和phi值
        float theta = x * step_x + *min_theta;
        float phi   = y * step_y + *min_phi;

        // 根据theta和phi计算世界坐标
        float3 P =
            make_float3(cyl.r * sinf(theta), cyl.r * phi, cyl.r * cosf(theta));

        // 投影世界坐标到图像像素
        float2 pixel;
        if (is_fisheye) {
            pixel = cam.projWorldToPixelFishEye(cyl.rotateVector_inv(P) +
                                                cyl.getCenter());
        } else {
            pixel =
                cam.projWorldToPixel(cyl.rotateVector_inv(P) + cyl.getCenter());
        }

        // 初始化颜色和掩码值
        uchar3 color = make_uchar3(0, 0, 0);
        uchar m      = 0;

        // 无掩码时处理像素
        if (mask == nullptr) {
            // 使用双线性插值获取源图像颜色
            if (pixel.x >= 0 && pixel.y >= 0 && pixel.x < src_w - 1 &&
                pixel.y < src_h - 1) {
                color = Bilinear(src_color, src_h, src_w, pixel);
            }

            // 计算在 cyl_image 中的位置
            int row = (phi - (*global_min_phi)) / step_y + 0.5;
            int col = (theta - (*global_min_theta)) / step_x + 0.5;
            if (x == 0 && y == 0) {
                uv[0] = 0;
                uv[1] = 0;
            }

            // 更新圆柱体图像
            if (row >= 0 && row < height && col >= 0 && col < width) {
                cyl_image[row * width + col] = color;
            }
        } else {   // 有掩码时处理像素
            // 使用双线性插值获取源图像颜色和掩码值
            if (pixel.x >= 0 && pixel.y >= 0 && pixel.x < src_w - 1 &&
                pixel.y < src_h - 1) {
                color = Bilinear(src_color, src_h, src_w, pixel);
                m = mask[(int)(pixel.y + 0.5f) * src_w + (int)(pixel.x + 0.5f)];
            }

            // 计算在 cyl_image 和 cyl_mask 中的位置
            int row = (phi - (*global_min_phi)) / step_y + 0.5;
            int col = (theta - (*global_min_theta)) / step_x + 0.5;
            if (x == 0 && y == 0) {
                uv[0] = 0;
                uv[1] = 0;
            }

            // 更新圆柱体图像和掩码
            if (row >= 0 && row < height && col >= 0 && col < width) {
                cyl_image[row * width + col] = color;
                cyl_mask[row * width + col]  = m;
            }
        }

        // 移动到下一个像素
        pixelIdx += total_thread;
    }
}

// static inline __device__ bool RayCylinderIntersection(float3 origin, float3 dir,
//                                                       float r,
//                                                       float3& intersection) {
//     float3 abs_dir = fabs(dir);

//     if (abs_dir.x < CLOSE_ZERO) {
//         intersection.x = origin.x;
//         intersection.z = sqrtf(powf(r, 2.0f) - powf(intersection.x, 2.0f));

//         if ((intersection.z - origin.z) / dir.z < 0)
//             intersection.z *= -1.0f;

//     } else if (abs_dir.z < CLOSE_ZERO) {
//         intersection.z = origin.z;
//         intersection.x = sqrtf(powf(r, 2.0f) - powf(intersection.z, 2.0f));

//         if ((intersection.x - origin.x) / dir.x < 0)
//             intersection.x *= -1.0f;

//     } else {
//         float k = dir.z / dir.x;
//         float b = origin.z - k * origin.x;

//         float check =
//             powf(2.0f * k * b, 2.0f) - 4 * (1 + k * k) * (b * b - r * r);
//         if (check < 0)
//             return false;

//         float item = sqrtf(check);

//         float x1 = (-2.0f * k * b + item) / (2 * (1 + k * k));
//         float x2 = (-2.0f * k * b - item) / (2 * (1 + k * k));

//         float z1 = k * x1 + b;
//         float z2 = k * x2 + b;

//         if ((x1 - origin.x) / dir.x > 0 && (z1 - origin.z) / dir.z > 0) {
//             intersection.x = x1;
//             intersection.z = z1;
//         } else {
//             intersection.x = x2;
//             intersection.z = z2;
//         }
//     }

//     float t_xz      = sqrtf(powf(intersection.x - origin.x, 2.0f) +
//                        powf(intersection.z - origin.z, 2.0f));
//     float tan_theta = dir.y / sqrtf(dir.x * dir.x + dir.z * dir.z);

//     intersection.y = origin.y + t_xz * tan_theta;
// }

// __global__ void GetBoundingBox_kernel(float* theta, float* phi, CylinderGPU cyl,
//                                       PinholeCameraGPU cam0,
//                                       PinholeCameraGPU cam1,
//                                       PinholeCameraGPU cam2, int height,
//                                       int width) {
//     int taskIdx = threadIdx.x + blockIdx.x * blockDim.x;

//     int x, y;
//     if (taskIdx < width) {
//         x = taskIdx;
//         y = 0;
//     } else if (taskIdx < 2 * width) {
//         x = taskIdx - width;
//         y = height - 1;
//     } else if (taskIdx < 2 * width + height) {
//         x = 0;
//         y = taskIdx - 2 * width;
//     } else if (taskIdx < 2 * (width + height)) {
//         x = width - 1;
//         y = taskIdx - (2 * width + height);
//     } else {
//         return;
//     }

//     float3 dir, origin, P;

//     if (blockIdx.y == 0) {
//         dir    = cyl.rotateVector(cam0.getRay(x, y));
//         origin = cyl.rotateVector(cam0.getCenter() - cyl.getCenter());
//     } else if (blockIdx.y == 1) {
//         dir    = cyl.rotateVector(cam1.getRay(x, y));
//         origin = cyl.rotateVector(cam1.getCenter() - cyl.getCenter());
//     } else {
//         dir    = cyl.rotateVector(cam2.getRay(x, y));
//         origin = cyl.rotateVector(cam2.getCenter() - cyl.getCenter());
//     }

//     RayCylinderIntersection(origin, dir, cyl.r, P);

//     theta[blockIdx.y * 2 * (width + height) + taskIdx] = atanf(P.x / P.z);
//     phi[blockIdx.y * 2 * (width + height) + taskIdx]   = P.y / cyl.r;
// }

// static __global__ void GetBoundingBox_kernel_v2(float* theta, float* phi,
//                                                 CylinderGPU cyl,
//                                                 PinholeCameraGPU cam0,
//                                                 PinholeCameraGPU cam1,
//                                                 PinholeCameraGPU cam2) {
//     int taskIdx      = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_thread = blockDim.x * gridDim.x;
//     int totalPixel   = cyl.offset[3];

//     while (taskIdx < totalPixel) {
//         int2 loc = cyl.boundary_pixel[taskIdx];
//         int x = loc.x, y = loc.y;

//         float3 dir, origin, P;
//         if (taskIdx < cyl.offset[1]) {
//             dir    = cyl.rotateVector(cam0.getRay(x, y));
//             origin = cyl.rotateVector(cam0.getCenter() - cyl.getCenter());
//         } else if (taskIdx < cyl.offset[2]) {
//             dir    = cyl.rotateVector(cam1.getRay(x, y));
//             origin = cyl.rotateVector(cam1.getCenter() - cyl.getCenter());
//         } else {
//             dir    = cyl.rotateVector(cam2.getRay(x, y));
//             origin = cyl.rotateVector(cam2.getCenter() - cyl.getCenter());
//         }
//         RayCylinderIntersection(origin, dir, cyl.r, P);

//         theta[taskIdx] = clamp(atan2f(P.x, P.z), -3.141592653 / 2 + 0.001,
//                                3.141592653 / 2 - 0.001);
//         phi[taskIdx]   = P.y / cyl.r;

//         taskIdx += total_thread;
//     }
// }

/**
 * 将四个通道的额外视图投影到圆柱图像
 *
 * @param extra_view 四个通道的额外视图，包含图像数据、掩码和视图尺寸等信息
 * @param extra_cyl_image 投影后的圆柱形图像，包含图像数据、掩码和UV坐标等信息
 * @param cylinder 圆柱体的参数和属性，如全局的theta和phi角等
 * @param cyl_image_width 圆柱图像的宽度
 * @param cyl_image_height 圆柱图像的高度
 * @return 总是返回true，表示函数执行成功
 */
bool proj4ChannelsExtraViewToCylinderImage_cuda(
    ViewGPU4Channels extra_view, CylinderImageGPU4Channels& extra_cyl_image,
    CylinderGPU& cylinder, int cyl_image_width, int cyl_image_height) {
    // 获取额外视图的高度和宽度，计算圆柱图像中通道的总数
    int height = extra_view.height, width = extra_view.width;
    int size_c = cylinder.offset[3];

    // 设置CUDA线程块和块的数量
    int num_thread = 512;
    int num_block  = min(65535, (size_c + num_thread - 1) / num_thread);
    int num_block2 =
        min(65535,
            (cyl_image_width * cyl_image_height + num_thread - 1) / num_thread);

    // 执行CUDA内核函数BackProjToSrc4Channels_kernel，将额外视图投影到圆柱图像
#if 1
    BackProjToSrc4Channels_kernel<<<num_block2, num_thread>>>(
        extra_view.image, extra_view.mask, extra_view.height, extra_view.width,
        cylinder, extra_view.camera, extra_cyl_image.image,
        extra_cyl_image.mask, extra_cyl_image.uv, cylinder.global_theta,
        cylinder.global_phi, cylinder.global_theta + 1, cylinder.global_phi + 1,
        cylinder.global_theta, cylinder.global_phi, cyl_image_height,
        cyl_image_width, false);

    // 检查CUDA执行是否出错
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
#endif
    return true;
}

// bool projExtraViewToCylinderImage_cuda(ViewGPU extra_view,
//                                        CylinderImageGPU& extra_cyl_image,
//                                        CylinderGPU& cylinder,
//                                        int cyl_image_width,
//                                        int cyl_image_height) {
//     int height = extra_view.height, width = extra_view.width;
//     int size_c     = cylinder.offset[3];
//     int num_thread = 512;
//     int num_block  = min(65535, (size_c + num_thread - 1) / num_thread);
//     int num_block2 =
//         min(65535,
//             (cyl_image_width * cyl_image_height + num_thread - 1) / num_thread);

//     BackProjToSrc_kernel<<<num_block2, num_thread>>>(
//         extra_view.image, extra_view.mask, extra_view.height, extra_view.width,
//         cylinder, extra_view.camera, extra_cyl_image.image,
//         extra_cyl_image.mask, extra_cyl_image.uv, cylinder.global_theta,
//         cylinder.global_phi, cylinder.global_theta + 1, cylinder.global_phi + 1,
//         cylinder.global_theta, cylinder.global_phi, cyl_image_height,
//         cyl_image_width, false);

//     checkCudaErrors(cudaDeviceSynchronize());
//     checkCudaErrors(cudaGetLastError());

//     return true;
// }

/**
 * 将多个视图的图像数据投影到一个圆柱体图像上。
 *
 * @param views 包含所有视图信息的向量，每个视图包含图像数据、掩码和相机参数。
 * @param cyl_images
 * 包含圆柱体图像信息的向量，每个圆柱体图像包括图像数据、掩码和UV坐标。
 * @param cylinder 包含圆柱体参数的对象，如偏移量、全局theta和phi值。
 * @param cyl_image_width 圆柱体图像的宽度。
 * @param cyl_image_height 圆柱体图像的高度。
 * @return 总是返回true，目前没有错误处理逻辑。
 *
 * 函数首先确定所有图像的尺寸，并根据圆柱体参数和图像尺寸计算线程块和块的数量。
 * 然后，使用CUDA kernel将每个视图的图像数据投影到对应的圆柱体图像上。
 * 最后，检查CUDA执行过程中是否有错误发生。
 */
bool projToCylinderImage_cuda(std::vector<ViewGPU> views,
                              std::vector<CylinderImageGPU>& cyl_images,
                              CylinderGPU& cylinder, int cyl_image_width,
                              int cyl_image_height) {
    // 确定图像尺寸并从圆柱体参数中获取一个尺寸信息
    int height = views[0].height, width = views[0].width;
    int size_c = cylinder.offset[3];
    // 计算执行CUDA kernel所需的线程块和块数
    int num_thread = 512;
    int num_block  = min(65535, (size_c + num_thread - 1) / num_thread);
    int num_block2 =
        min(65535,
            (cyl_image_width * cyl_image_height + num_thread - 1) / num_thread);

    // 检查之前是否有CUDA错误
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 对三个视图的数据进行投影操作
    for (int i = 0; i < 3; i++) {
        BackProjToSrc_kernel<<<num_block2, num_thread>>>(
            views[i].image, views[i].mask, views[i].height, views[i].width,
            cylinder, views[i].camera, cyl_images[i].image, cyl_images[i].mask,
            cyl_images[i].uv, cylinder.global_theta, cylinder.global_phi,
            cylinder.global_theta + 1, cylinder.global_phi + 1,
            cylinder.global_theta, cylinder.global_phi, cyl_image_height,
            cyl_image_width, true);
    }

    // 再次检查CUDA执行过程中是否有错误
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    return true;
}

// __global__ void
// ForwardProjToCylinder_kernel(float2* cylinder_coor,   // theta phi
//                              CylinderGPU cyl, PinholeCameraGPU cam, int height,
//                              int width) {
//     int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
//     int total_thread = blockDim.x * gridDim.x;
//     int totalPixel   = height * width;
//     while (pixelIdx < totalPixel) {
//         int x = pixelIdx % width;
//         int y = pixelIdx / width;

//         float3 dir    = cyl.rotateVector(cam.getRay(x, y));
//         float3 origin = cyl.rotateVector(cam.getCenter() - cyl.getCenter());

//         float3 P;
//         RayCylinderIntersection(origin, dir, cyl.r, P);

//         cylinder_coor[pixelIdx].x = atanf(P.x / P.z);
//         cylinder_coor[pixelIdx].y = P.y / cyl.r;

//         pixelIdx += total_thread;
//     }
// }
