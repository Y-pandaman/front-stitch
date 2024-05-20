#include "render/seam_finder.cuh"

/**
 * 在GPU上计算三个圆柱图像之间的差异。
 *
 * @param left 左侧圆柱图像
 * @param mid 中间圆柱图像
 * @param right 右侧圆柱图像
 * @param diff 存储图像差异的结果数组
 * @param height 图像的高度
 * @param width 图像的宽度
 *
 * 此函数通过逐个像素比较，计算出左右两个图像与中间图像的差异，
 * 并将差异值存储在diff数组中。
 */
__global__ void color_diff_kernel(CylinderImageGPU left, CylinderImageGPU mid,
                                  CylinderImageGPU right, DIFF_TYPE* diff,
                                  int height, int width) {
    // 计算当前线程处理的像素索引和线程总数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 遍历所有像素，计算差异
    while (pixelIdx < totalPixel) {
        // 计算像素的x和y坐标
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 获取三个图像在当前像素点的RGB值
        uchar3 left_rgb  = left.getImageValue(x, y);
        uchar3 mid_rgb   = mid.getImageValue(x, y);
        uchar3 right_rgb = right.getImageValue(x, y);

        // 获取三个图像在当前像素点的掩码值
        uchar left_mask  = left.getMaskValue(x, y);
        uchar mid_mask   = mid.getMaskValue(x, y);
        uchar right_mask = right.getMaskValue(x, y);

        // 将RGB值从无符号字符转换为浮点数，并除以MASK_MAX
        float3 l = make_float3((float)left_rgb.x, (float)left_rgb.y,
                               (float)left_rgb.z) /
                   MASK_MAX;
        float3 m =
            make_float3((float)mid_rgb.x, (float)mid_rgb.y, (float)mid_rgb.z) /
            MASK_MAX;
        float3 r = make_float3((float)right_rgb.x, (float)right_rgb.y,
                               (float)right_rgb.z) /
                   MASK_MAX;

        // 初始化差异值
        float diff_lm = 0.05f, diff_mr = 0.05;
        // 计算左右和中间图像，以及中间和右侧图像的差异
        if (left_mask == 255 && mid_mask == 255)
            diff_lm = fabs(l.x - m.x) + fabs(l.y - m.y) + fabs(l.z - m.z);
        if (right_mask == 255 && mid_mask == 255)
            diff_mr = fabs(r.x - m.x) + fabs(r.y - m.y) + fabs(r.z - m.z);
        // 将两个差异值相加，并转换为DIFF_TYPE类型，存储在diff数组中
        diff[pixelIdx] = (diff_lm + diff_mr) * DIFF_TYPE_MAX_VALUE;

        // 更新像素索引，准备处理下一个像素
        pixelIdx += total_thread;
    }
}

/**
 * 在GPU上执行掩码操作的内核函数。
 * 对三个圆柱图像的掩码进行合并和分类，计算总掩码、仅在左侧的掩码、仅在右侧的掩码以及重叠区域的掩码。
 *
 * @param left 输入的左圆柱图像的掩码。
 * @param mid 输入的中圆柱图像的掩码。
 * @param right 输入的右圆柱图像的掩码。
 * @param total_mask 输出的总掩码，包含所有输入掩码的或操作结果。
 * @param left_only_mask 输出掩码，表示只在左图像中存在的像素。
 * @param right_only_mask 输出掩码，表示只在右图像中存在的像素。
 * @param overlap_mask 输出掩码，表示在所有输入图像中都存在的像素。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 */
__global__ void mask_op_kernel(CylinderImageGPU left, CylinderImageGPU mid,
                               CylinderImageGPU right, uchar* total_mask,
                               uchar* left_only_mask, uchar* right_only_mask,
                               uchar* overlap_mask, int height, int width) {
    // 计算每个线程处理的像素索引和线程总数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;   // 图像总像素数

    // 遍历所有像素，执行掩码操作
    while (pixelIdx < totalPixel) {
        // 计算像素的x和y坐标
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        // 获取三个输入图像在当前像素点的掩码值，并标准化到[0, MASK_MAX]之间
        uchar l = left.getMaskValue(x, y) / MASK_MAX;
        uchar m = mid.getMaskValue(x, y) / MASK_MAX;
        uchar r = right.getMaskValue(x, y) / MASK_MAX;

        // 计算总掩码、仅在左侧的掩码、仅在右侧的掩码以及重叠区域的掩码
        uchar total   = l | m | r;
        uchar l_only  = (1 - l * m) * l;
        uchar r_only  = (1 - m * r) * r;
        l_only        = l_only - (l_only * r_only);
        uchar overlap = l * m * r;

        // 将计算得到的掩码值写入对应的输出数组
        total_mask[pixelIdx]      = total * MASK_MAX;
        left_only_mask[pixelIdx]  = l_only;
        right_only_mask[pixelIdx] = r_only;
        overlap_mask[pixelIdx]    = overlap;

        // 更新处理的像素索引，准备处理下一个像素
        pixelIdx += total_thread;
    }
}

/**
 * 在GPU上计算缝合掩码。
 *
 * @param left 左侧图像的GPU表示。
 * @param mid 中间图像的GPU表示。
 * @param right 右侧图像的GPU表示。
 * @param left_mask_only 左侧图像的独有掩码。
 * @param total_mask 总体掩码。
 * @param right_mask_only 右侧图像的独有掩码。
 * @param separate_line 分割线数组，用于指定中间图像的宽度。
 * @param seam_mask_left 左侧缝合掩码的输出数组。
 * @param seam_mask_mid 中间缝合掩码的输出数组。
 * @param seam_mask_right 右侧缝合掩码的输出数组。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 *
 * 此函数通过遍历图像的每个像素，根据给定的条件计算并更新缝合掩码。
 */
__global__ void get_seam_mask(CylinderImageGPU left, CylinderImageGPU mid,
                              CylinderImageGPU right,
                              const uchar* left_mask_only,
                              const uchar* total_mask,
                              const uchar* right_mask_only,
                              const int* separate_line, uchar* seam_mask_left,
                              uchar* seam_mask_mid, uchar* seam_mask_right,
                              int height, int width) {
    // 计算当前线程处理的像素索引和总线程数
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    // 遍历所有像素
    while (pixelIdx < totalPixel) {
        // 计算像素的x和y坐标
        int x = pixelIdx % width;
        int y = pixelIdx / width;
        // 获取中间图像在当前像素行的左右边界x坐标
        int x1 = separate_line[y];
        int x2 = separate_line[y + height];

        // 根据像素是否只属于左侧或右侧图像，设置相应的缝合掩码值
        if (left_mask_only[pixelIdx]) {
            seam_mask_left[pixelIdx] = MASK_MAX;
        } else if (right_mask_only[pixelIdx]) {
            seam_mask_right[pixelIdx] = MASK_MAX;
        } else {
            // 计算像素属于哪个图像，并设置相应的缝合掩码
            if (x < x1)
                seam_mask_left[pixelIdx] = left.getMaskValue(x, y);
            else if (x < x2)
                seam_mask_mid[pixelIdx] = mid.getMaskValue(x, y);
            else
                seam_mask_right[pixelIdx] = right.getMaskValue(x, y);
        }

        // 更新中间图像的缝合掩码，确保总掩码减去左右两侧的掩码值
        seam_mask_mid[pixelIdx] = total_mask[pixelIdx] -
                                  seam_mask_left[pixelIdx] -
                                  seam_mask_right[pixelIdx];

        // 移动到下一个像素
        pixelIdx += total_thread;
    }
}

/**
 * 在CUDA平台上执行缝合查找的函数。
 *
 * @param cylImages 存储了三个CylinderImageGPU对象的向量，表示待处理的图像数据。
 * @param height 图像的高度。
 * @param width 图像的宽度。
 * @param separate_line
 * 一个二维向量，存储了前一次缝合的结果，用于指导当前的缝合计算。
 * @param range 影响缝合计算的像素范围。
 * @param seam_masks 存储三个掩膜指针的向量，用于标识图像中的缝合路径。
 * @param is_first 标记是否是第一次调用该函数，用于融合多次计算的结果。
 * @return 总是返回true，表示函数执行成功。
 */
__host__ bool SeamFind_cuda(std::vector<CylinderImageGPU> cylImages, int height,
                            int width,
                            std::vector<std::vector<int>>& separate_line,
                            int range, std::vector<uchar*>& seam_masks,
                            bool is_first) {
    assert(cylImages.size() == 3);   // 确保输入的图像数量正确

    int numPixel   = height * width;   // 计算图像的总像素数
    int num_thread = 512;              // 设置每个线程块的线程数
    int num_block  = min(
        65535, (numPixel + num_thread - 1) /
                   num_thread);   // 计算所需的线程块数量，以确保所有像素被处理

    // 分配CUDA设备内存，用于存储掩膜和差异数据
    uchar *_left_mask_only, *_total_mask, *_right_mask_only, *_overlap_mask;
    cudaMalloc((void**)&_left_mask_only, sizeof(uchar) * numPixel);
    cudaMalloc((void**)&_total_mask, sizeof(uchar) * numPixel);
    cudaMalloc((void**)&_right_mask_only, sizeof(uchar) * numPixel);
    cudaMalloc((void**)&_overlap_mask, sizeof(uchar) * numPixel);

    DIFF_TYPE* _diff;
    cudaMalloc((void**)&_diff, sizeof(DIFF_TYPE) * numPixel);
    DIFF_TYPE* diff;
    cudaHostAlloc(
        (void**)&diff, sizeof(DIFF_TYPE) * numPixel,
        cudaHostAllocDefault);   // 分配主机内存，用于存放从设备内存拷贝的差异数据

    int* _separate_line;
    cudaMalloc((void**)&_separate_line,
               sizeof(int) * height * 2);   // 存储分离线的设备内存分配

    // 执行掩膜操作和颜色差异计算的CUDA内核调用
    mask_op_kernel<<<num_block, num_thread>>>(
        cylImages[0], cylImages[1], cylImages[2], _total_mask, _left_mask_only,
        _right_mask_only, _overlap_mask, height, width);

    color_diff_kernel<<<num_block, num_thread>>>(
        cylImages[0], cylImages[1], cylImages[2], _diff, height, width);

    cudaMemcpy(
        diff, _diff, sizeof(DIFF_TYPE) * numPixel,
        cudaMemcpyDeviceToHost);   // 将设备内存中的差异数据拷贝到主机内存

    std::vector<std::vector<int>>
        new_separate_line;   // 用于存储新的分离线计算结果
    new_separate_line.resize(2);

    // 执行缝合线搜索
    search_seam(diff, range, separate_line[0], new_separate_line[0], height,
                width);
    search_seam(diff, range, separate_line[1], new_separate_line[1], height,
                width);

    // 如果不是第一次调用，则融合新旧缝合线结果
    if (is_first == false) {
        for (int i = 0; i < separate_line[0].size(); ++i) {
            new_separate_line[0][i] = 0.5 * (float)separate_line[0][i] +
                                      0.5 * (float)new_separate_line[0][i];
            new_separate_line[1][i] = 0.5 * (float)separate_line[1][i] +
                                      0.5 * (float)new_separate_line[1][i];
        }
    }

    // 将新的分离线结果拷贝回设备内存
    cudaMemcpy(_separate_line, new_separate_line[0].data(),
               sizeof(int) * height, cudaMemcpyHostToDevice);
    cudaMemcpy(_separate_line + height, new_separate_line[1].data(),
               sizeof(int) * height, cudaMemcpyHostToDevice);

    separate_line = new_separate_line;   // 更新分离线向量

    // 执行掩膜生成的CUDA内核调用
    get_seam_mask<<<num_block, num_thread>>>(
        cylImages[0], cylImages[1], cylImages[2], _left_mask_only, _total_mask,
        _right_mask_only, _separate_line, seam_masks[0], seam_masks[1],
        seam_masks[2], height, width);

    // 释放所有申请的内存资源
    cudaFree(_left_mask_only);
    cudaFree(_total_mask);
    cudaFree(_right_mask_only);
    cudaFree(_overlap_mask);
    cudaFree(_diff);
    cudaFreeHost(diff);
    cudaFree(_separate_line);

    return true;   // 表示函数执行成功
}
