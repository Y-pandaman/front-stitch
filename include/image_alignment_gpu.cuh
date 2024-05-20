#ifndef GALAHOLOSYSTEM_IMG_ALIGNMENT_CUDA_H
#define GALAHOLOSYSTEM_IMG_ALIGNMENT_CUDA_H

#include <opencv2/cudafilters.hpp>
#include <opencv2/opencv.hpp>
#include <vector_functions.h>
#include <vector_types.h>

#define USE_TEXTURE_MUM 0

class GNSolver;

class ImageAlignmentCUDA {
public:
    ImageAlignmentCUDA(int img_rows, int img_cols, int grid_num_rows,
                       int grid_num_cols);
    ~ImageAlignmentCUDA();

    void SetSrcTargetImgs(float* d_src_img, int src_rows, int src_cols,
                          float* d_target_img, int target_rows, int target_cols,
                          int2 src_min_uv, int2 target_min_uv, int gpu_idx = 0);
    void SetSrcTargetImgsFromHost(cv::Mat& src_img, cv::Mat& src_mask,
                                  int2 src_min_uv, cv::Mat& target_img,
                                  cv::Mat& target_mask, int2 target_min_uv);
    void SetSrcTargetImgsFromDevice(uchar3* d_src_img, uchar* d_src_mask,
                                    int src_rows, int src_cols,
                                    uchar3* d_target_img, uchar* d_target_mask,
                                    int target_rows, int target_cols);
    void AlignSrcImgToTarget(int iter_num = 10);
    void ShowNodeImg(std::vector<std::vector<float2> >& node_vec_vis);
    void ShowComparisonImg();
    void WarpSrcImg(uchar3* d_warped_src_img, uchar* d_warped_src_mask,
                    uchar3* d_src_img, uchar* d_src_mask, int src_rows,
                    int src_cols);
    void WarpSrcImg();

public:
    cv::Ptr<cv::cuda::Filter> gauss_filter_;
    cv::Ptr<cv::cuda::Filter> dx_scharr_filter_, dy_scharr_filter_;

    float *d_src_img_buffer_ = nullptr, *d_target_img_buffer_ = nullptr;

    cv::cuda::GpuMat d_src_gray_img_, d_target_gray_img_;
    cv::cuda::GpuMat d_dy_img_, d_dx_img_;
    cv::cuda::GpuMat d_warped_src_gray_img_;
    int img_rows_, img_cols_, node_img_rows_, node_img_cols_;

    float2 *d_node_vec_, *d_original_node_vec_;
    int* d_pixel_rela_idx_vec_;
    float* d_pixel_rela_weight_vec_;
    int* d_node_rela_idx_vec_;
    float* d_node_rela_weight_vec_;

    int2 src_min_uv_, target_min_uv_;

    float triangle_u_, triangle_v_;
    GNSolver* gn_solver_;

    cudaStream_t streams_[4];
    std::vector<cv::cuda::Stream> streams_array_;

    uchar3* image_buffer_ = nullptr;
    uchar* mask_buffer_;
};

#endif   // GALAHOLOSYSTEM_IMG_ALIGNMENT_CUDA_H
