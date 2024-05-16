#include "shaders.h"

float* FrameBuffer::getCudaPtr() {
    if (!cuda_res_) {
        printf("cuda_res = %d\n", cuda_res_);
        return nullptr;
    }
    return m_dev_ptr;
}

/**
 * 更新CUDA内存。
 * 此函数将与CUDA图形资源相关联的内存更新到设备指针指向的内存中。
 * 
 * @return 总是返回true，表示内存更新成功。
 */
bool FrameBuffer::updateCudaMem() {
    cudaError_t cErr;
    cudaArray_t cArray;

    // 映射图形资源到CUDA
    cErr = cudaGraphicsMapResources(1, &cuda_res_);

    // 获取映射后的数组资源
    cErr = cudaGraphicsSubResourceGetMappedArray(&cArray, cuda_res_, 0, 0);
    
    // 从数组中拷贝数据到设备内存
    cErr = cudaMemcpyFromArray(m_dev_ptr, cArray, 0, 0, size_in_byte,
                               cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    
    // 解除资源映射
    cErr = cudaGraphicsUnmapResources(1, &cuda_res_);

    return true;
}
