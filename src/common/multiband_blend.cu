#include "common/multiband_blend.cuh"
#include "util/cuda_utils.h"

__device__ __constant__ float GauKernel[25] = {
    0.0039, 0.0156, 0.0234, 0.0156, 0.0039, 0.0156, 0.0625, 0.0938, 0.0625,
    0.0156, 0.0234, 0.0938, 0.1406, 0.0938, 0.0234, 0.0156, 0.0625, 0.0938,
    0.0625, 0.0156, 0.0039, 0.0156, 0.0234, 0.0156, 0.0039};

/**
 * @brief 对给定颜色通道进行加权乘积计算。
 *
 * 该模板函数针对一个特定的颜色通道，将源图像的像素值与给定权重相乘，
 * 并累加到对应的颜色值和权重和中。
 *
 * @tparam CHANNEL 颜色通道的数量。
 * @param color 指向存储颜色值的数组的指针。
 * @param src 指向源图像像素值的数组的指针。
 * @param aw 参与累加的权重总和的引用。
 * @param w 当前像素的权重。
 */
template <int CHANNEL>
__device__ static inline void weight_product(float* color, short* src,
                                             float& aw, float w) {
#pragma unroll
    // 遍历颜色通道，对每个通道的色彩值进行加权累加
    for (int i = 0; i < CHANNEL; i++) {
        color[i] = color[i] + (float)src[i] * w;
    }
    // 累加权重
    aw += w;
}

__global__ void fast_erode(uchar* seam_mask, uchar* mask, int radius,
                           int height, int width) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;
    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        int remain = 1;

        for (int i = -radius; i <= radius && remain == 1; i++) {
            for (int j = -radius; j <= radius; j++) {
                int u = x + j, v = y + i;

                if (u < 0 || v < 0 || u >= width || v >= height)
                    continue;

                if (mask[u + v * width] == 0) {
                    remain = 0;
                    break;
                }
            }
        }

        seam_mask[pixelIdx] = seam_mask[pixelIdx] * remain;

        pixelIdx += total_thread;
    }
}

/**
 * @brief
 * 核心函数，用于将短整型三维数据转换回无符号字符三维数据，并应用掩码处理。
 * @param src 指向短整型三维数据的指针。
 * @param dst 指向无符号字符三维数据的指针，为转换结果。
 * @param mask0 指向第一个掩码数组的指针。
 * @param mask1 指向第二个掩码数组的指针。
 * @param mask2 指向第三个掩码数组的指针。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 该函数通过线程块和网格的组织方式，对输入的短整型图像数据进行并行处理，转换为无符号字符类型，
 * 并应用三个掩码数组来决定每个像素是否被处理。处理过程中，将短整型数据的范围从-32768到32767，
 * 转换为无符号字符的范围0到255，并且根据掩码值决定像素是否被最终输出。
 */
__global__ void convertBack_kernel(short3* src, uchar3* dst, uchar* mask0,
                                   uchar* mask1, uchar* mask2, int height,
                                   int width) {
    // 计算当前线程处理的像素索引和总计线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;   // 图像总像素数

    // 循环处理所有像素
    while (pixelIdx < totalPixel) {
        // 根据三个掩码计算有效像素的标志
        int remian = mask0[pixelIdx] / 255 | mask1[pixelIdx] / 255 |
                     mask2[pixelIdx] / 255;

        // 对每个颜色通道进行转换和掩码应用
        dst[pixelIdx].x = remian *
                          clamp((float)src[pixelIdx].x, 0.0f, 32767.0f) /
                          32767.0f * 255.0f;
        dst[pixelIdx].y = remian *
                          clamp((float)src[pixelIdx].y, 0.0f, 32767.0f) /
                          32767.0f * 255.0f;
        dst[pixelIdx].z = remian *
                          clamp((float)src[pixelIdx].z, 0.0f, 32767.0f) /
                          32767.0f * 255.0f;

        // 更新处理的像素索引
        pixelIdx += total_thread;
    }
}

/**
 * 模板函数，用于将给定通道数的源图像数据转换为指定数据类型，并应用缩放因子和掩码。
 *
 * @tparam CHANNEL 源图像的通道数。
 * @param src 源图像数据的指针。
 * @param mask 应用到源图像的掩码，用于指定某些像素是否要被处理。
 * @param dst 转换后图像数据的指针。
 * @param scale 缩放因子，用于调整源图像数据的值。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 函数在GPU上并行执行，将源图像数据（ uchar 类型）转换为 short
 * 类型，并应用缩放。
 * 如果源图像为三通道且掩码对应像素值为0，则将该像素所有通道的值设置为-32768。
 */
template <int CHANNEL>
__global__ void converTo_kernel(uchar* src, uchar* mask, short* dst,
                                float scale, int height, int width) {
    // 每个线程处理的像素索引
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    // 总线程数
    int total_thread = blockDim.x * gridDim.x;
    // 图像总像素数
    int totalPixel = height * width;

    // 循环处理所有像素
    while (pixelIdx < totalPixel) {
#pragma unroll
        // 遍历当前像素的所有通道，应用缩放因子
        for (int i = 0; i < CHANNEL; i++) {
            dst[CHANNEL * pixelIdx + i] =
                (float)src[CHANNEL * pixelIdx + i] * scale;
        }

        // 如果是三通道图像且掩码值为0，则将像素值设置为-32768
        if (CHANNEL == 3 && mask[pixelIdx] == 0) {
#pragma unroll
            for (int i = 0; i < CHANNEL; i++) {
                dst[CHANNEL * pixelIdx + i] = -32768;
            }
        }

        // 更新处理的像素索引
        pixelIdx += total_thread;
    }
}

/*
ALGORITHM description
The function performs the downsampling step of the Gaussian pyramid
construction. First, it convolves the source image with the kernel: GauKernel
Then, it downsamples the image by rejecting even rows and columns.

Each thread is responsible for each pixel in high level (low resolution).
*/
/**
 * @brief 下采样金字塔 kernel 函数
 *
 * 该模板函数用于对给定的图像进行下采样，创建一个低分辨率的图像版本。
 * 它通过应用高斯滤波器来平均相邻像素，从而降低图像的分辨率。
 *
 * @tparam CHANNEL 图像的通道数
 * @param src 输入图像的指针（短整型）
 * @param dst 输出图像的指针（短整型）
 * @param height 输入图像的高度
 * @param width 输入图像的宽度
 *
 * 注意：该函数假设输入图像的尺寸是可被 2 整除的。
 */
template <int CHANNEL>
__global__ void PyrDown_kernel(short* src, short* dst, int height, int width) {
    // 计算当前线程处理的像素索引和总计数线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 循环处理所有像素
    while (pixelIdx < totalPixel) {
        // 计算输入像素的坐标和索引
        int src_x        = 2 * (pixelIdx % width);
        int src_y        = 2 * (pixelIdx / width);
        int src_pixelIdx = src_y * (width * 2) + src_x;

        // 初始化颜色数组和权重
        float color[CHANNEL];
        for (int i = 0; i < CHANNEL; i++)
            color[i] = 0;
        float weight = 0.0f;

        // 应用 5x5 高斯滤波器
#pragma unroll
        for (int i = 0; i < 5; i++) {
#pragma unroll
            for (int j = 0; j < 5; j++) {
                int u = src_x + j - 2;
                int v = src_y + i - 2;

                // 忽略越界的像素
                if (u < 0 || v < 0 || u >= width * 2 || v >= height * 2)
                    continue;

                // 忽略值为 -32768 的无效像素
                if (src[CHANNEL * (v * 2 * width + u)] == -32768)
                    continue;

                // 计算像素的加权颜色值
                weight_product<CHANNEL>(color,
                                        src + CHANNEL * (v * 2 * width + u),
                                        weight, GauKernel[i * 5 + j]);
            }
        }

        // 根据权重平均颜色值，并存储到输出图像中
#pragma unroll
        if (weight == 0) {
            // 如果权重为 0，则将像素设置为无效值 -32768
            for (int i = 0; i < CHANNEL; i++) {
                dst[CHANNEL * pixelIdx + i] = -32768;
            }
        } else {
            // 如果权重不为 0，则将加权颜色值存储到输出图像中
            for (int i = 0; i < CHANNEL; i++) {
                color[i]                    = color[i] / weight;
                dst[CHANNEL * pixelIdx + i] = (short)color[i];
            }
        }

        // 更新处理的像素索引
        pixelIdx += total_thread;
    }
}

/*
ALGORITHM description
The image is first upsized to twice the original in each dimension, with the
new (even) rows filled with 0s. Thereafter, a convolution is performed with
Gaussian filter to approximate the value of the "missing" pixels.

This filter is also normalized to 4, rather rhan 1. (because inserted rows have
0s)

Each thread is responsible for each pixel in low level (high resolution).
*/
/**
 * @brief PyrUp操作的CUDA内核函数，用于将输入图像放大两倍。
 *
 * @tparam SIGN 一个符号标志，用于控制放大操作是增加像素值还是减去像素值。
 * @param src 输入图像的短整型三维数组。
 * @param dst 输出图像的短整型三维数组。
 * @param height 输入图像的高度。
 * @param width 输入图像的宽度。
 *
 * 该函数使用了5x5的高斯核进行放大操作。每个输出像素由输入图像中附近25个像素经过加权平均得到。
 * 根据SIGN的值，可以实现放大像素值（SIGN=1）或减小像素值（SIGN=-1）的效果。
 */
template <int SIGN>
__global__ void PyrUp_kernel(short3* src, short3* dst, int height, int width) {
    // 计算当前线程处理的像素索引和总计线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 循环处理所有像素
    while (pixelIdx < totalPixel) {
        // 计算当前处理的像素的x和y坐标
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 初始化颜色向量和权重
        float3 color = make_float3(0, 0, 0);
        float weight = 0;

        // 使用双重循环应用5x5高斯核进行加权平均
#pragma unroll
        for (int i = 0; i < 5; i++) {
#pragma unroll
            for (int j = 0; j < 5; j++) {
                int u = x + j - 2;
                int v = y + i - 2;

                // 忽略越界的像素
                if (u < 0 || v < 0 || u >= width || v >= height)
                    continue;

                // 忽略无效像素（值为-32768）
                if (src[v / 2 * width / 2 + u / 2].x == -32768)
                    continue;

                // 计算当前像素的加权贡献
                if (u % 2 == 0 && v % 2 == 0) {
                    weight_product<3>(
                        &color.x, (short*)(src + (v / 2 * width / 2 + u / 2)),
                        weight, 4 * GauKernel[i * 5 + j]);
                }
            }
        }

        // 如果权重不为0，则更新像素值
        if (weight != 0) {
            // 根据权重调整颜色值
            color = color / weight;

            // 根据SIGN标志调整像素值
            color.x = (float)dst[pixelIdx].x + SIGN * color.x;
            color.y = (float)dst[pixelIdx].y + SIGN * color.y;
            color.z = (float)dst[pixelIdx].z + SIGN * color.z;

            // 确保像素值在合法范围内
            color = clamp(color, -32768.0f, 32767.0f);
        }

        // 更新输出图像的像素值
        dst[pixelIdx].x = color.x;
        dst[pixelIdx].y = color.y;
        dst[pixelIdx].z = color.z;

        // 移动到下一个像素
        pixelIdx += total_thread;
    }
}

/**
 * 在GPU上执行加权融合操作。
 *
 * @param I0 输入图像1的像素值，类型为short3。
 * @param mask0 输入图像1的掩码，类型为short，用于指定像素的权重。
 * @param I1 输入图像2的像素值，类型为short3。
 * @param mask1 输入图像2的掩码，类型为short，用于指定像素的权重。
 * @param I2 输入图像3的像素值，类型为short3。
 * @param mask2 输入图像3的掩码，类型为short，用于指定像素的权重。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 该函数通过加权融合三个输入图像的像素值来生成一个新的图像（通过修改I1参数实现），
 * 其中每个像素的权重由对应的mask参数指定。权重值被归一化到[0,1]范围内。
 * 无效的像素（总权重为0）在输出图像中被设为(0,0,0)。
 */

__global__ void WeightedBlend_kernel(short3* I0, short* mask0, short3* I1,
                                     short* mask1, short3* I2, short* mask2,
                                     int height, int width) {
    // 计算当前线程处理的像素索引，以及所有线程能够处理的像素总数。
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 遍历所有像素，执行加权融合操作。
    while (pixelIdx < totalPixel) {
        // 计算当前像素的行列坐标。
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 将掩码值转换为浮点数，并归一化到[0,1]范围。
        float w0 = (float)mask0[pixelIdx] / 32767.0f;
        float w1 = (float)mask1[pixelIdx] / 32767.0f;
        float w2 = (float)mask2[pixelIdx] / 32767.0f;
        // 计算所有输入图像的总权重。
        float total_w = w0 + w1 + w2;

        // 如果总权重大于0，执行加权融合计算，否则将像素值设为0。
        if (total_w > 0) {
            I1[pixelIdx].x = (short)(((float)I0[pixelIdx].x * w0 +
                                      (float)I1[pixelIdx].x * w1 +
                                      (float)I2[pixelIdx].x * w2) /
                                     total_w);
            I1[pixelIdx].y = (short)(((float)I0[pixelIdx].y * w0 +
                                      (float)I1[pixelIdx].y * w1 +
                                      (float)I2[pixelIdx].y * w2) /
                                     total_w);
            I1[pixelIdx].z = (short)(((float)I0[pixelIdx].z * w0 +
                                      (float)I1[pixelIdx].z * w1 +
                                      (float)I2[pixelIdx].z * w2) /
                                     total_w);
        } else {
            // 如果总权重为0，将像素值设为0。
            I1[pixelIdx].x = 0;
            I1[pixelIdx].y = 0;
            I1[pixelIdx].z = 0;
        }

        // 更新处理的像素索引，准备处理下一个像素。
        pixelIdx += total_thread;
    }
}

/**
 * 生成拉普拉斯金字塔
 *
 * @param img 输入的圆柱图像，类型为CylinderImageGPU
 * @param seam_mask 缝隙掩码，指向uchar类型的指针
 * @param levels 金字塔的级别数
 * @return 返回一个包含各级别图像的std::vector<SeamImageGPU>
 */
std::vector<SeamImageGPU> LaplacianPyramid(CylinderImageGPU img,
                                           uchar* seam_mask, int levels) {
    std::vector<SeamImageGPU> pyr_imgs;   // 存储金字塔图像的容器

    int num_thread = 512;                         // 每个块的线程数
    int height = img.height, width = img.width;   // 输入图像的高度和宽度
    // 计算块的数量，以适应CUDA线程块的限制
    int num_block = min(65535, (height * width + num_thread - 1) / num_thread);

    // 分配CUDA内存以存储当前图像和掩码
    short3* current_img;
    cudaMalloc((void**)&current_img, sizeof(short3) * height * width);
    short* current_mask;
    cudaMalloc((void**)&current_mask, sizeof(short) * height * width);

    // 将输入图像和掩码转换为中间格式
    converTo_kernel<3><<<num_block, num_thread>>>(
        (uchar*)img.image, img.mask, (short*)current_img, 32767.0f / 255.0f,
        height, width);
    // 如果seam_mask非空，将其转换为中间格式
    converTo_kernel<1><<<num_block, num_thread>>>(
        seam_mask, nullptr, current_mask, 32767.0f / 255.0f, height, width);

    // 将转换后的图像和掩码作为第一级金字塔图像添加到容器中
    pyr_imgs.emplace_back(
        SeamImageGPU(current_img, current_mask, height, width));

    // 生成金字塔的剩余级别
    for (int i = 1; i < levels; i++) {
        // 更新图像高度和宽度以生成下一级别
        height /= 2;
        width /= 2;
        // 计算适应新图像尺寸的块数量
        num_block = min(65535, (height * width + num_thread - 1) / num_thread);

        // 分配内存以存储下一级别的图像和掩码
        short3* _down_scale_img;
        cudaMalloc((void**)&_down_scale_img, sizeof(short3) * height * width);
        short* _down_scale_mask;
        cudaMalloc((void**)&_down_scale_mask, sizeof(short) * height * width);

        // 对当前图像和掩码进行下采样
        PyrDown_kernel<3><<<num_block, num_thread>>>(
            (short*)current_img, (short*)_down_scale_img, height, width);
        PyrDown_kernel<1><<<num_block, num_thread>>>(
            current_mask, _down_scale_mask, height, width);

        // 将下采样的图像和掩码添加到金字塔图像容器中
        pyr_imgs.emplace_back(
            SeamImageGPU(_down_scale_img, _down_scale_mask, height, width));

        // 更新当前图像和掩码指针以指向下一级别的图像和掩码
        current_img  = _down_scale_img;
        current_mask = _down_scale_mask;
    }

    // 从倒数第二级到第一级进行上采样，以重构除最高级别外的所有级别图像
    for (int i = 0; i < levels - 1; i++) {
        int num_block = min(
            65535, (pyr_imgs[i].height * pyr_imgs[i].width + num_thread - 1) /
                       num_thread);
        // 进行上采样
        PyrUp_kernel<-1><<<num_block, num_thread>>>(
            pyr_imgs[i + 1].image, pyr_imgs[i].image, pyr_imgs[i].height,
            pyr_imgs[i].width);
    }

    return pyr_imgs;   // 返回构建的拉普拉斯金字塔图像容器
}

/**
 * 将三个图像金字塔进行加权融合，生成一个新的图像金字塔。
 *
 * @param pyr_img0 第一个输入图像金字塔，作为融合基础。
 * @param pyr_img1 第二个输入图像金字塔，参与融合。
 * @param pyr_img2 第三个输入图像金字塔，参与融合。
 * @return 返回一个融合后的图像金字塔，存储在std::vector中。
 */
std::vector<SeamImageGPU> BlendPyramid(std::vector<SeamImageGPU> pyr_img0,
                                       std::vector<SeamImageGPU> pyr_img1,
                                       std::vector<SeamImageGPU> pyr_img2) {
    int num_thread = 512;   // 设置每个block中的线程数

    std::vector<SeamImageGPU> blendedPyramid;   // 存储融合后的图像金字塔

    // 遍历图像金字塔的每一层，进行加权融合
    for (int i = 0; i < pyr_img0.size(); i++) {
        // 计算执行融合操作所需的block数量
        int num_block = min(
            65535, (pyr_img0[i].height * pyr_img0[i].width + num_thread - 1) /
                       num_thread);
        // 调用CUDA kernel函数进行加权融合
        WeightedBlend_kernel<<<num_block, num_thread>>>(
            pyr_img0[i].image, pyr_img0[i].mask, pyr_img1[i].image,
            pyr_img1[i].mask, pyr_img2[i].image, pyr_img2[i].mask,
            pyr_img0[i].height, pyr_img0[i].width);

        // 将融合后的图像添加到结果金字塔中
        blendedPyramid.emplace_back(SeamImageGPU(
            pyr_img1[i].image, nullptr, pyr_img1[i].height, pyr_img1[i].width));
    }

    return blendedPyramid;
}

/**
 * 将金字塔图像序列合并回单个图像。
 * 该函数首先通过上采样操作将金字塔的每一层（从倒数第二层开始）合并到其上一层，
 * 最后将合并后的金字塔底层图像转换回初始的圆柱图像格式。
 *
 * @param blendedPyramid
 * 包含已融合图像的金字塔。这些图像经过之前的处理，准备进行上采样合并。
 * @param cylImages 包含三个圆柱图像和对应掩码的向量，用于最后的图像转换。
 */

void CollapsePyramid(std::vector<SeamImageGPU> blendedPyramid,
                     std::vector<CylinderImageGPU> cylImages) {
    int num_thread = 512, num_block = 0;

    // 从倒数第二层开始，对每一层图像进行上采样，合并到其上一层。
    for (int i = blendedPyramid.size() - 2; i >= 0; i--) {
        // 根据当前层的图像大小计算线程块和线程数。
        num_block =
            min(65535, (blendedPyramid[i].height * blendedPyramid[i].width +
                        num_thread - 1) /
                           num_thread);
        // 调用PyrUp_kernel核函数，进行上采样合并操作。
        PyrUp_kernel<1><<<num_block, num_thread>>>(
            blendedPyramid[i + 1].image, blendedPyramid[i].image,
            blendedPyramid[i].height, blendedPyramid[i].width);
    }

    // 计算用于最终图像转换的线程块和线程数。
    num_block = min(65535, (blendedPyramid[0].height * blendedPyramid[0].width +
                            num_thread - 1) /
                               num_thread);
    // 调用convertBack_kernel核函数，将金字塔底层图像转换回圆柱图像格式。
    convertBack_kernel<<<num_block, num_thread>>>(
        blendedPyramid[0].image, cylImages[1].image, cylImages[0].mask,
        cylImages[1].mask, cylImages[2].mask, blendedPyramid[0].height,
        blendedPyramid[0].width);
}

void Erode_seam_mask(std::vector<CylinderImageGPU> cylImages,
                     std::vector<uchar*> seam_masks, int radius) {
    int num_thread = 512;

    for (int i = 0; i < 3; i++) {
        int h = cylImages[i].height, w = cylImages[i].width;
        int num_block = min(65535, (h * w + num_thread - 1) / num_thread);
        fast_erode<<<num_block, num_thread>>>(seam_masks[i], cylImages[i].mask,
                                              radius, h, w);
    }
}

/**
 * 在CUDA设备上执行多带融合操作。
 *
 * @param cylImages 包含待融合的三个圆柱体图像的向量。
 * @param seam_masks 包含三个图像的接缝掩膜的向量，用于指定融合区域。
 */
__host__ void MultiBandBlend_cuda(std::vector<CylinderImageGPU> cylImages,
                                  std::vector<uchar*> seam_masks) {
    // 创建CUDA事件用于测量执行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    int levels = 5;   // 设定拉普拉斯金字塔的层数

    // 为每张输入图像构建拉普拉斯金字塔
    std::vector<SeamImageGPU> pyr_img0 =
        LaplacianPyramid(cylImages[0], seam_masks[0], levels);
    std::vector<SeamImageGPU> pyr_img1 =
        LaplacianPyramid(cylImages[1], seam_masks[1], levels);
    std::vector<SeamImageGPU> pyr_img2 =
        LaplacianPyramid(cylImages[2], seam_masks[2], levels);

    // 将三幅图像的金字塔融合成一个金字塔
    std::vector<SeamImageGPU> blendedPyramid =
        BlendPyramid(pyr_img0, pyr_img1, pyr_img2);

    // 从融合的金字塔中恢复最终的融合图像
    CollapsePyramid(blendedPyramid, cylImages);

    // 确保所有CUDA操作都已完成，检查有无错误发生
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 释放拉普拉斯金字塔中分配的内存资源
    for (int i = 0; i < pyr_img0.size(); ++i) {
        pyr_img0[i].free();
        pyr_img1[i].free();
        pyr_img2[i].free();
    }
}