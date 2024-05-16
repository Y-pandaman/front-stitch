#include "image_alignment_cpu.h"
#include "innoreal_timer.hpp"
#include "math_utils.h"
#include <Eigen/Eigen>
#include <fstream>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

/**
 * 计算双线性插值的系数
 *
 * 该函数用于根据给定的坐标和区间边界计算双线性插值所需的四个系数。
 * 这些系数可用于从四个角点的值插值出任意点的值。
 *
 * @param coef_00 双线性插值系数00（对应左上角点）
 * @param coef_10 双线性插值系数10（对应右上角点）
 * @param coef_01 双线性插值系数01（对应左下角点）
 * @param coef_11 双线性插值系数11（对应右下角点）
 * @param x 待插值点的x坐标
 * @param y 待插值点的y坐标
 * @param x_0 x坐标的左边界
 * @param y_0 y坐标的上边界
 * @param x_1 x坐标的右边界
 * @param y_1 y坐标的下边界
 */
__device__ __host__ __forceinline__ void
CalcBilinearCoefs(float& coef_00, float& coef_10, float& coef_01,
                  float& coef_11, float x, float y, float x_0, float y_0,
                  float x_1, float y_1) {
    // 计算x和y方向的归一化系数
    float norm_coef_x = 1 / (x_1 - x_0);
    float norm_coef_y = 1 / (y_1 - y_0);

    // 计算x和y方向的插值系数
    float coef_x_0 = (x_1 - x) * norm_coef_x;
    float coef_y_0 = (y_1 - y) * norm_coef_y;

    // 计算四个双线性插值系数
    coef_00 = coef_x_0 * coef_y_0;
    coef_10 = (1 - coef_x_0) * coef_y_0;
    coef_01 = coef_x_0 * (1 - coef_y_0);
    coef_11 = (1 - coef_x_0) * (1 - coef_y_0);
}

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
    if (val_00 > 10.0f || val_00 < -10.0f)
        return 65535.0f;
    float val_10 = img.at<float>(y_0, x_1);
    if (val_10 > 10.0f || val_10 < -10.0f)
        return 65535.0f;
    float val_01 = img.at<float>(y_1, x_0);
    if (val_01 > 10.0f || val_01 < -10.0f)
        return 65535.0f;
    float val_11 = img.at<float>(y_1, x_1);
    if (val_11 > 10.0f || val_11 < -10.0f)
        return 65535.0f;

    return coef_00 * val_00 + coef_01 * val_01 + coef_10 * val_10 +
           coef_11 * val_11;
}

inline cv::Vec3f GetPixelValueBilinear(cv::Mat_<cv::Vec3f>& img, float x,
                                       float y) {
    int x_0 = (int)x;
    int y_0 = (int)y;
    int x_1 = x_0 + 1;
    int y_1 = y_0 + 1;

    float coef_x_0 = x_1 - x;
    float coef_y_0 = y_1 - y;

    if (x_0 < 0 || y_0 < 0 || x_0 >= img.cols || y_0 >= img.rows || x_1 < 0 ||
        y_1 < 0 || x_1 >= img.cols || y_1 >= img.rows)
        return cv::Vec3f(65525.0f, 0.0f, 0.0f);   // invalid

    float coef_00 = coef_x_0 * coef_y_0;
    float coef_10 = (1 - coef_x_0) * coef_y_0;
    float coef_01 = coef_x_0 * (1 - coef_y_0);
    float coef_11 = (1 - coef_x_0) * (1 - coef_y_0);

    cv::Vec3f val_00 = img(y_0, x_0);
    cv::Vec3f val_10 = img(y_0, x_1);
    cv::Vec3f val_01 = img(y_1, x_0);
    cv::Vec3f val_11 = img(y_1, x_1);

    return coef_00 * val_00 + coef_01 * val_01 + coef_10 * val_10 +
           coef_11 * val_11;
}

static void Get4NearestNode(std::vector<int>& pixel_rela_idx_vec,
                            std::vector<float>& pixel_rela_weight_vec, float x,
                            float y, int grid_width, int grid_height,
                            int node_cols) {
    int node_row = y / grid_height;
    int node_col = x / grid_width;
    pixel_rela_idx_vec.push_back(node_row * node_cols + node_col);
    pixel_rela_idx_vec.push_back(node_row * node_cols + node_col + 1);
    pixel_rela_idx_vec.push_back((node_row + 1) * node_cols + node_col);
    pixel_rela_idx_vec.push_back((node_row + 1) * node_cols + node_col + 1);
    float coef_00, coef_10, coef_01, coef_11;
    CalcBilinearCoefs(coef_00, coef_10, coef_01, coef_11, x, y,
                      node_col * grid_width, node_row * grid_height,
                      (node_col + 1) * grid_width,
                      (node_row + 1) * grid_height);
    pixel_rela_weight_vec.push_back(coef_00);   // top-left
    pixel_rela_weight_vec.push_back(coef_10);   // top-right
    pixel_rela_weight_vec.push_back(coef_01);   // bottom-left
    pixel_rela_weight_vec.push_back(coef_11);   // bottom-right
}

ImageAlignmentCPU::ImageAlignmentCPU(int img_rows, int img_cols,
                                     int grid_img_rows, int grid_img_cols)
    : img_rows_(img_rows), img_cols_(img_cols), grid_img_rows_(grid_img_rows),
      grid_img_cols_(grid_img_cols), node_rows_(grid_img_rows + 1),
      node_cols_(grid_img_cols + 1) {
    /** 向上取整，使得所有的node覆盖整幅图片 */
    grid_width_  = (img_cols_ + grid_img_cols - 1) / grid_img_cols;
    grid_height_ = (img_rows_ + grid_img_rows - 1) / grid_img_rows;

    node_vec_.clear();
    pixel_rela_idx_vec_.clear();
    pixel_rela_weight_vec_.clear();
    node_rela_idx_vec_.clear();
    node_rela_weight_vec_.clear();

    /**
     * 计算全图的节点位置，以及像素到节点的关系和插值权重。
     * 该过程首先遍历图像网格，为每个网格节点记录其在图像中的2D位置，
     * 然后对于每个像素，确定它所属的四个节点，并计算从该像素到这四个节点的插值权重。
     */
    for (int row = 0, node_pos_row = 0; row < node_rows_;
         ++row, node_pos_row += grid_height_) {
        for (int col = 0, node_pos_col = 0; col < node_cols_;
             ++col, node_pos_col += grid_width_) {
            // 为每个网格节点记录其在图像中的位置
            node_vec_.push_back(make_float2(node_pos_col, node_pos_row));
        }
    }

    float coef_00, coef_10, coef_01, coef_11;
    for (int row = 0; row < img_rows; ++row) {
        for (int col = 0; col < img_cols; ++col) {
            // 确定当前像素所属的四个节点
            int node_row = row / grid_height_;
            int node_col = col / grid_width_;
            pixel_rela_idx_vec_.push_back(node_row * node_cols_ + node_col);
            pixel_rela_idx_vec_.push_back(node_row * node_cols_ + node_col + 1);
            pixel_rela_idx_vec_.push_back((node_row + 1) * node_cols_ +
                                          node_col);
            pixel_rela_idx_vec_.push_back((node_row + 1) * node_cols_ +
                                          node_col + 1);

            // 计算当前像素到所属四个节点的插值权重
            CalcBilinearCoefs(coef_00, coef_10, coef_01, coef_11, col, row,
                              node_col * grid_width_, node_row * grid_height_,
                              (node_col + 1) * grid_width_,
                              (node_row + 1) * grid_height_);
            // 记录插值权重
            pixel_rela_weight_vec_.push_back(coef_00);   // top-left
            pixel_rela_weight_vec_.push_back(coef_10);   // top-right
            pixel_rela_weight_vec_.push_back(coef_01);   // bottom-left
            pixel_rela_weight_vec_.push_back(coef_11);   // bottom-right
        }
    }

    /**
     * 此代码块构建了一个基于图像坐标的节点关系索引、权重以及各个项的预留容量。
     * 它首先通过两层循环遍历图像网格中的节点，为每一对相邻节点建立关系，
     * 并记录其在图像坐标系中的u,v偏移量和权重。然后，为后续计算的各种项（特征匹配项、
     * 光学流项、零点项、正则化项）的残差和三元组预留足够的内存空间。
     */

    /** 构建节点之间的相对索引、u,v向量和权重向量，适用于图像的上半部分区域 */
    for (int row = 0; row < node_rows_ - 1; ++row) {
        for (int col = 0; col < node_cols_ - 1; ++col) {
            // 添加节点之间的相对索引
            node_rela_idx_vec_.push_back(row * node_cols_ + col);
            node_rela_idx_vec_.push_back(row * node_cols_ + col + 1);
            node_rela_idx_vec_.push_back((row + 1) * node_cols_ + col + 1);
            // 添加节点的u,v偏移量，此处假设v方向向下，故为负值
            node_u_vec_.push_back(0.0f);
            node_v_vec_.push_back(-grid_width_ / (float)grid_height_);
            // 添加节点之间的权重
            node_rela_weight_vec_.push_back(1.0f);
        }
    }

    /** 构建节点之间的相对索引、u,v向量和权重向量，适用于图像的下半部分区域 */
    for (int row = 1; row < node_rows_; ++row) {
        for (int col = 1; col < node_cols_; ++col) {
            // 添加节点之间的相对索引
            node_rela_idx_vec_.push_back(row * node_cols_ + col);
            node_rela_idx_vec_.push_back(row * node_cols_ + col - 1);
            node_rela_idx_vec_.push_back((row - 1) * node_cols_ + col - 1);
            // 添加节点的u,v偏移量，此处假设v方向向下，故为负值
            node_u_vec_.push_back(0.0f);
            node_v_vec_.push_back(-grid_width_ / (float)grid_height_);
            // 添加节点之间的权重
            node_rela_weight_vec_.push_back(1.0f);
        }
    }

    /** 为后续计算的各个项预留内存空间，以减少动态分配的开销 */
    feat_corr_term_triple_vec_.reserve(img_rows_ * img_cols_);
    feat_corr_term_residual_vec_.reserve(img_rows_ * img_cols_);
    photo_term_triple_vec_.reserve(img_rows_ * img_cols_);
    photo_term_residual_vec_.reserve(img_rows_ * img_cols_);
    zero_term_triple_vec_.reserve(img_rows_ * img_cols_);
    zero_term_residual_vec_.reserve(img_rows_ * img_cols_);
    reg_term_triple_vec_.reserve(img_rows_ * img_cols_);
    reg_term_residual_vec_.reserve(img_rows_ * img_cols_);
}

ImageAlignmentCPU::~ImageAlignmentCPU() { }

void ImageAlignmentCPU::ShowNodeImg(
    std::vector<std::vector<float2> >& node_vec_vis) {
    cv::Mat warped_node_img = cv::Mat::zeros(img_rows_, img_cols_, CV_8UC3);
    for (int j = 0; j < node_vec_vis.front().size(); ++j) {
        cv::circle(
            warped_node_img,
            cv::Point(node_vec_vis.front()[j].x, node_vec_vis.front()[j].y), 3,
            cv::Scalar(143, 42, 157), 3, cv::LINE_AA);
    }
    for (int j = 0; j < node_vec_vis.back().size(); ++j) {
        cv::circle(
            warped_node_img,
            cv::Point(node_vec_vis.back()[j].x, node_vec_vis.back()[j].y), 3,
            cv::Scalar(81, 231, 111), 3, cv::LINE_AA);
    }
    cv::imshow("warped_src_img", warped_node_img);
    cv::waitKey(1);
}

void ImageAlignmentCPU::ShowComparisonImg() {
    cv::Mat total_src_image    = cv::Mat::zeros(img_rows_, img_cols_, CV_32FC1);
    cv::Mat total_target_image = cv::Mat::zeros(img_rows_, img_cols_, CV_32FC1);
    cv::Mat diff_image         = cv::Mat::zeros(img_rows_, img_cols_, CV_32FC1);
    for (int row = 0; row < src_img_.rows; ++row) {
        for (int col = 0; col < src_img_.cols; ++col) {
            if (warped_src_img_.at<float>(row, col) > 10.0f)
                warped_src_img_.at<float>(row, col) = 0.0f;
            total_src_image.at<float>(row, col) =
                warped_src_img_.at<float>(row, col);
        }
    }
    for (int row = 0; row < target_img_.rows; ++row) {
        for (int col = 0; col < target_img_.cols; ++col) {
            if (target_img_.at<float>(row, col) > 10.0f)
                target_img_.at<float>(row, col) = 0.0f;
            total_target_image.at<float>(row + target_min_uv_.y,
                                         col + target_min_uv_.x) =
                target_img_.at<float>(row, col);
        }
    }
    for (int row = 0; row < diff_image.rows; ++row) {
        for (int col = 0; col < diff_image.cols; ++col) {
            diff_image.at<float>(row, col) =
                fabs(total_src_image.at<float>(row, col) -
                     total_target_image.at<float>(row, col));
        }
    }
    cv::imshow("diff_image", diff_image * 1);
    cv::waitKey(0);
}

void ImageAlignmentCPU::WarpSrcImg(cv::Mat_<cv::Vec3f>& warped_src_image,
                                   cv::Mat_<cv::Vec3f>& src_image) {
    /** backward mapping */
    int x_0, x_1, y_0, y_1;
    float coef_00, coef_10, coef_01, coef_11;
    int tl, tr, bl, br;
    for (int pixel_idx = 0; pixel_idx < pixel_rela_idx_vec_.size() / 4;
         ++pixel_idx) {
        int row = pixel_idx / img_cols_;
        int col = pixel_idx % img_cols_;
        tl      = 4 * pixel_idx;
        tr      = 4 * pixel_idx + 1;
        bl      = 4 * pixel_idx + 2;
        br      = 4 * pixel_idx + 3;

        float2 warped_pixel =
            pixel_rela_weight_vec_[tl] * node_vec_[pixel_rela_idx_vec_[tl]] +
            pixel_rela_weight_vec_[tr] * node_vec_[pixel_rela_idx_vec_[tr]] +
            pixel_rela_weight_vec_[bl] * node_vec_[pixel_rela_idx_vec_[bl]] +
            pixel_rela_weight_vec_[br] * node_vec_[pixel_rela_idx_vec_[br]];

        float2 src_image_uv  = warped_pixel - src_min_uv_;
        int2 target_image_uv = make_int2(col, row);

        if (src_image_uv.y >= 0 && src_image_uv.y < src_image.rows &&
            src_image_uv.x >= 0 && src_image_uv.x < src_image.cols &&
            target_image_uv.y >= 0 && target_image_uv.y < src_img_.rows &&
            target_image_uv.x >= 0 && target_image_uv.x < src_img_.cols) {
            cv::Vec3f src_intensity = GetPixelValueBilinear(
                src_image, src_image_uv.x, src_image_uv.y);
            warped_src_image.at<cv::Vec3f>(target_image_uv.y,
                                           target_image_uv.x) = src_intensity;
        }
    }

    cv::imshow("warped_src_image", warped_src_image);
    cv::waitKey(0);
}

void ImageAlignmentCPU::WarpSrcImg() {
    std::cout << src_img_.rows << " : " << src_img_.cols << std::endl;
    std::cout << warped_src_img_.rows << " : " << warped_src_img_.cols
              << std::endl;
    std::cout << target_img_.rows << " : " << target_img_.cols << std::endl;
    std::cout << img_rows_ << " : " << img_cols_ << std::endl;
    std::cout << "----------------" << std::endl;

    /** backward mapping */
    warped_src_img_ = cv::Mat::zeros(img_rows_, img_cols_, CV_32FC1);
    int x_0, x_1, y_0, y_1;
    float coef_00, coef_10, coef_01, coef_11;
    int tl, tr, bl, br;
    for (int pixel_idx = 0; pixel_idx < pixel_rela_idx_vec_.size() / 4;
         ++pixel_idx) {
        int row = pixel_idx / img_cols_;
        int col = pixel_idx % img_cols_;
        tl      = 4 * pixel_idx;
        tr      = 4 * pixel_idx + 1;
        bl      = 4 * pixel_idx + 2;
        br      = 4 * pixel_idx + 3;

        float2 warped_pixel =
            pixel_rela_weight_vec_[tl] * node_vec_[pixel_rela_idx_vec_[tl]] +
            pixel_rela_weight_vec_[tr] * node_vec_[pixel_rela_idx_vec_[tr]] +
            pixel_rela_weight_vec_[bl] * node_vec_[pixel_rela_idx_vec_[bl]] +
            pixel_rela_weight_vec_[br] * node_vec_[pixel_rela_idx_vec_[br]];

        float2 src_image_uv  = warped_pixel - src_min_uv_;
        int2 target_image_uv = make_int2(col, row);

        if (src_image_uv.y >= 0 && src_image_uv.y < src_img_.rows &&
            src_image_uv.x >= 0 && src_image_uv.x < src_img_.cols &&
            target_image_uv.y >= 0 && target_image_uv.y < src_img_.rows &&
            target_image_uv.x >= 0 && target_image_uv.x < src_img_.cols) {
            float src_intensity =
                GetPixelValueBilinear(src_img_, src_image_uv.x, src_image_uv.y);
            warped_src_img_.at<float>(target_image_uv.y, target_image_uv.x) =
                src_intensity;
        }
    }

    cv::imshow("src_img", src_img_);
    cv::imshow("warped_src_img", warped_src_img_);
    cv::waitKey(0);
}

void ImageAlignmentCPU::SetSrcTargetImgs(cv::Mat& src_img, cv::Mat& src_mask,
                                         int2 src_min_uv, cv::Mat& target_img,
                                         cv::Mat& target_mask,
                                         int2 target_min_uv,
                                         std::vector<float4> match_corr) {
    match_corr_ = match_corr;

    /** src_img和target_img被歸一化到[0,1]，特別定義invalid爲65535.0，這樣gradient的邊緣的invalid被定義爲大於10.0 */
    cv::cvtColor(src_img, src_img_, CV_RGB2GRAY);
    cv::cvtColor(target_img, target_img_, CV_RGB2GRAY);
    src_img_.convertTo(src_img_, CV_32FC1, 1.0 / 255);
    target_img_.convertTo(target_img_, CV_32FC1, 1.0 / 255);

    for (int row = 0; row < src_img.rows; ++row) {
        for (int col = 0; col < src_img.cols; ++col) {
            if (src_mask.at<uchar>(row, col) < 128) {
                src_img_.at<float>(row, col) = 65525.0f;
            }
        }
    }
    for (int row = 0; row < target_img.rows; ++row) {
        for (int col = 0; col < target_img.cols; ++col) {
            if (target_mask.at<uchar>(row, col) < 128) {
                target_img_.at<float>(row, col) = 65525.0f;
            }
        }
    }
    src_min_uv_    = make_float2(src_min_uv.x, src_min_uv.y);
    target_min_uv_ = make_float2(target_min_uv.x, target_min_uv.y);

    cv::GaussianBlur(src_img_, blur_src_img_, cv::Size(5, 5), 2, 2);
    cv::GaussianBlur(target_img_, blur_target_img_, cv::Size(5, 5), 2, 2);

#if 1
    cv::Mat temp_src_img    = src_img.clone();
    cv::Mat temp_target_img = target_img.clone();
    for (int i = 0; i < match_corr_.size(); ++i) {
        std::cout << cv::Point(match_corr_[i].x, match_corr_[i].y) << std::endl;
        cv::circle(
            temp_src_img, cv::Point(match_corr_[i].x, match_corr_[i].y), 1,
            cv::Scalar((1257 * i) % 255, (17 * i) % 255, (157 * i) % 255), 3,
            cv::LINE_AA);
        cv::circle(
            temp_target_img, cv::Point(match_corr_[i].z, match_corr_[i].w), 1,
            cv::Scalar((1257 * i) % 255, (17 * i) % 255, (157 * i) % 255), 3,
            cv::LINE_AA);
    }
    cv::imshow("src_img", temp_src_img);
    cv::imshow("target_img", temp_target_img);
    cv::waitKey(0);
#endif

    cv::Mat kx_horizontal, ky_horizontal, kx_vertical, ky_vertical;
    dy_img_ = cv::Mat_<float>::zeros(blur_src_img_.size());
    dx_img_ = cv::Mat_<float>::zeros(blur_src_img_.size());
    cv::getDerivKernels(kx_horizontal, ky_horizontal, 1, 0, CV_SCHARR, true);
    cv::getDerivKernels(kx_vertical, ky_vertical, 0, 1, CV_SCHARR, true);
    cv::sepFilter2D(blur_src_img_, dy_img_, dy_img_.depth(), kx_vertical,
                    ky_vertical);
    cv::sepFilter2D(blur_src_img_, dx_img_, dx_img_.depth(), kx_horizontal,
                    ky_horizontal);
    cv::imshow("dx_img", dx_img_ * 100);
    cv::imshow("dy_img", dy_img_ * 100);
    cv::waitKey(1);
}

/** node建立在target圖上，warp的時候使用backward mapping */
void ImageAlignmentCPU::AlignSrcImgToTarget(int sample_step) {
    std::vector<float2> original_node_vec = node_vec_;
    std::vector<std::vector<float2> > node_vec_vis;

    for (int iter = 0; iter < 15; ++iter) {
        printf("iter: %d\n", iter);
        WarpSrcImg();
        ShowComparisonImg();
        float feat_corr_term_w = 0.5f, photo_term_w = 1.0f, reg_term_w = 0.2f,
              zero_term_w = 0.001f;
        innoreal::InnoRealTimer timer;
        timer.TimeStart();

        feat_corr_term_triple_vec_.resize(0);
        feat_corr_term_residual_vec_.resize(0);
        photo_term_triple_vec_.resize(0);
        photo_term_residual_vec_.resize(0);
        zero_term_triple_vec_.resize(0);
        zero_term_residual_vec_.resize(0);
        reg_term_triple_vec_.resize(0);
        reg_term_residual_vec_.resize(0);

        std::vector<Eigen::Triplet<float> > feat_corr_term_triple_vec;
        std::vector<float> feat_corr_term_residual_vec;
        int feat_corr_term_cnt = 0;
        for (int i = 0; i < match_corr_.size(); ++i) {
            float2 src_uv    = make_float2(match_corr_[i].x, match_corr_[i].y);
            float2 target_uv = make_float2(match_corr_[i].z, match_corr_[i].w);
            std::vector<int> feat_pixel_rela_idx_vec;
            std::vector<float> feat_pixel_rela_weight_vec;
            Get4NearestNode(feat_pixel_rela_idx_vec, feat_pixel_rela_weight_vec,
                            target_uv.x, target_uv.y, grid_width_, grid_height_,
                            node_cols_);
            float2 warped_pixel = feat_pixel_rela_weight_vec[0] *
                                      node_vec_[feat_pixel_rela_idx_vec[0]] +
                                  feat_pixel_rela_weight_vec[1] *
                                      node_vec_[feat_pixel_rela_idx_vec[1]] +
                                  feat_pixel_rela_weight_vec[2] *
                                      node_vec_[feat_pixel_rela_idx_vec[2]] +
                                  feat_pixel_rela_weight_vec[3] *
                                      node_vec_[feat_pixel_rela_idx_vec[3]];
            warped_pixel = warped_pixel + target_min_uv_ - src_min_uv_;

            feat_corr_term_residual_vec.push_back(feat_corr_term_w *
                                                  (warped_pixel.x - src_uv.x));
            feat_corr_term_residual_vec.push_back(feat_corr_term_w *
                                                  (warped_pixel.y - src_uv.y));

            feat_corr_term_triple_vec.push_back(Eigen::Triplet<float>(
                feat_corr_term_cnt, feat_pixel_rela_idx_vec[0] * 2,
                feat_corr_term_w * feat_pixel_rela_weight_vec[0]));
            feat_corr_term_triple_vec.push_back(Eigen::Triplet<float>(
                feat_corr_term_cnt + 1, feat_pixel_rela_idx_vec[0] * 2 + 1,
                feat_corr_term_w * feat_pixel_rela_weight_vec[0]));

            feat_corr_term_triple_vec.push_back(Eigen::Triplet<float>(
                feat_corr_term_cnt, feat_pixel_rela_idx_vec[1] * 2,
                feat_corr_term_w * feat_pixel_rela_weight_vec[1]));
            feat_corr_term_triple_vec.push_back(Eigen::Triplet<float>(
                feat_corr_term_cnt + 1, feat_pixel_rela_idx_vec[1] * 2 + 1,
                feat_corr_term_w * feat_pixel_rela_weight_vec[1]));

            feat_corr_term_triple_vec.push_back(Eigen::Triplet<float>(
                feat_corr_term_cnt, feat_pixel_rela_idx_vec[2] * 2,
                feat_corr_term_w * feat_pixel_rela_weight_vec[2]));
            feat_corr_term_triple_vec.push_back(Eigen::Triplet<float>(
                feat_corr_term_cnt + 1, feat_pixel_rela_idx_vec[2] * 2 + 1,
                feat_corr_term_w * feat_pixel_rela_weight_vec[2]));

            feat_corr_term_triple_vec.push_back(Eigen::Triplet<float>(
                feat_corr_term_cnt, feat_pixel_rela_idx_vec[3] * 2,
                feat_corr_term_w * feat_pixel_rela_weight_vec[3]));
            feat_corr_term_triple_vec.push_back(Eigen::Triplet<float>(
                feat_corr_term_cnt + 1, feat_pixel_rela_idx_vec[3] * 2 + 1,
                feat_corr_term_w * feat_pixel_rela_weight_vec[3]));

            feat_corr_term_cnt += 2;
        }
        Eigen::SparseMatrix<float> feat_corr_term_jacobian(
            feat_corr_term_cnt, node_rows_ * node_cols_ * 2);
        feat_corr_term_jacobian.setFromTriplets(
            feat_corr_term_triple_vec.begin(), feat_corr_term_triple_vec.end());
        Eigen::VectorXf feat_corr_term_residual(
            feat_corr_term_residual_vec.size());
        memcpy(feat_corr_term_residual.data(),
               feat_corr_term_residual_vec.data(),
               feat_corr_term_residual_vec.size() * sizeof(float));

        /** 計算photo_term的jacobian和residual，[xy,xy,xy,xy...] */
        std::vector<Eigen::Triplet<float> > photo_term_triple_vec;
        std::vector<float> photo_term_residual_vec;
        int photo_term_cnt = 0;
        int tl, tr, bl, br;
        for (int pixel_idx = 0; pixel_idx < pixel_rela_idx_vec_.size() / 4;
             ++pixel_idx) {
            int row = pixel_idx / img_cols_;
            int col = pixel_idx % img_cols_;

            if (row < img_rows_ && col < img_cols_) {
                tl = 4 * pixel_idx;
                tr = 4 * pixel_idx + 1;
                bl = 4 * pixel_idx + 2;
                br = 4 * pixel_idx + 3;

                float2 warped_pixel = pixel_rela_weight_vec_[tl] *
                                          node_vec_[pixel_rela_idx_vec_[tl]] +
                                      pixel_rela_weight_vec_[tr] *
                                          node_vec_[pixel_rela_idx_vec_[tr]] +
                                      pixel_rela_weight_vec_[bl] *
                                          node_vec_[pixel_rela_idx_vec_[bl]] +
                                      pixel_rela_weight_vec_[br] *
                                          node_vec_[pixel_rela_idx_vec_[br]];
                float2 src_image_uv    = warped_pixel - src_min_uv_;
                float2 target_image_uv = make_float2(col, row) - target_min_uv_;
                float src_intensity    = GetPixelValueBilinear(
                    blur_src_img_, src_image_uv.x, src_image_uv.y);
                float dy_val = GetPixelValueBilinear(dy_img_, src_image_uv.x,
                                                     src_image_uv.y);
                float dx_val = GetPixelValueBilinear(dx_img_, src_image_uv.x,
                                                     src_image_uv.y);
                float target_intensity = GetPixelValueBilinear(
                    blur_target_img_, target_image_uv.x, target_image_uv.y);

                if (target_intensity > 10.0f || src_intensity > 10.0f ||
                    fabs(dy_val) > 10.0f || fabs(dx_val) > 10.0f)
                    continue;

                photo_term_residual_vec.push_back(
                    photo_term_w * (src_intensity - target_intensity));

                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec_[tl] * 2,
                    photo_term_w * dx_val * pixel_rela_weight_vec_[tl]));
                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec_[tl] * 2 + 1,
                    photo_term_w * dy_val * pixel_rela_weight_vec_[tl]));

                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec_[tr] * 2,
                    photo_term_w * dx_val * pixel_rela_weight_vec_[tr]));
                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec_[tr] * 2 + 1,
                    photo_term_w * dy_val * pixel_rela_weight_vec_[tr]));

                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec_[bl] * 2,
                    photo_term_w * dx_val * pixel_rela_weight_vec_[bl]));
                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec_[bl] * 2 + 1,
                    photo_term_w * dy_val * pixel_rela_weight_vec_[bl]));

                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec_[br] * 2,
                    photo_term_w * dx_val * pixel_rela_weight_vec_[br]));
                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec_[br] * 2 + 1,
                    photo_term_w * dy_val * pixel_rela_weight_vec_[br]));

                ++photo_term_cnt;
            }
        }

        Eigen::SparseMatrix<float> photo_term_jacobian(
            photo_term_cnt, node_rows_ * node_cols_ * 2);
        photo_term_jacobian.setFromTriplets(photo_term_triple_vec.begin(),
                                            photo_term_triple_vec.end());
        Eigen::VectorXf photo_term_residual(photo_term_residual_vec.size());
        memcpy(photo_term_residual.data(), photo_term_residual_vec.data(),
               photo_term_residual_vec.size() * sizeof(float));

        /** 計算zero_term的jacobian和residual，[xy,xy,xy,xy...] */
        std::vector<Eigen::Triplet<float> > zero_term_triple_vec;
        std::vector<float> zero_term_residual_vec;
        int zero_term_cnt = 0;
        for (int node_idx = 0; node_idx < node_vec_.size(); ++node_idx) {
            zero_term_residual_vec.push_back(
                zero_term_w *
                (node_vec_[node_idx].x - original_node_vec[node_idx].x));
            zero_term_residual_vec.push_back(
                zero_term_w *
                (node_vec_[node_idx].y - original_node_vec[node_idx].y));

            zero_term_triple_vec.push_back(Eigen::Triplet<float>(
                zero_term_cnt, node_idx * 2, zero_term_w * 1));
            zero_term_triple_vec.push_back(Eigen::Triplet<float>(
                zero_term_cnt + 1, node_idx * 2 + 1, zero_term_w * 1));

            zero_term_cnt += 2;
        }
        Eigen::SparseMatrix<float> zero_term_jacobian(
            zero_term_cnt, node_rows_ * node_cols_ * 2);
        zero_term_jacobian.setFromTriplets(zero_term_triple_vec.begin(),
                                           zero_term_triple_vec.end());
        Eigen::VectorXf zero_term_residual(zero_term_residual_vec.size());
        memcpy(zero_term_residual.data(), zero_term_residual_vec.data(),
               zero_term_residual_vec.size() * sizeof(float));

        /** 計算reg_term的jacobian和residual，[xy,xy,xy,xy...] */
        std::vector<Eigen::Triplet<float> > reg_term_triple_vec;
        std::vector<float> reg_term_residual_vec;
        int reg_term_cnt = 0;
        for (int triangle_idx = 0; triangle_idx < node_rela_idx_vec_.size() / 3;
             ++triangle_idx) {
            int v1_idx       = node_rela_idx_vec_[3 * triangle_idx];
            int v2_idx       = node_rela_idx_vec_[3 * triangle_idx + 1];
            int v3_idx       = node_rela_idx_vec_[3 * triangle_idx + 2];
            float2 v1        = node_vec_[v1_idx];
            float2 v2        = node_vec_[v2_idx];
            float2 v3        = node_vec_[v3_idx];
            float weight     = node_rela_weight_vec_[triangle_idx];
            float constant_u = node_u_vec_[triangle_idx];
            float constant_v = node_v_vec_[triangle_idx];

            reg_term_residual_vec.push_back(
                reg_term_w * (v1.x - (v2.x + constant_u * (v3.x - v2.x) +
                                      constant_v * (v3.y - v2.y))));
            reg_term_residual_vec.push_back(
                reg_term_w * (v1.y - (v2.y + constant_u * (v3.y - v2.y) +
                                      constant_v * (-v3.x + v2.x))));

            reg_term_triple_vec.push_back(Eigen::Triplet<float>(
                reg_term_cnt, v1_idx * 2, reg_term_w * 1));
            reg_term_triple_vec.push_back(Eigen::Triplet<float>(
                reg_term_cnt, v2_idx * 2, reg_term_w * (-1 + constant_u)));
            reg_term_triple_vec.push_back(Eigen::Triplet<float>(
                reg_term_cnt, v3_idx * 2, reg_term_w * (-constant_u)));
            reg_term_triple_vec.push_back(Eigen::Triplet<float>(
                reg_term_cnt, v3_idx * 2 + 1, reg_term_w * (-constant_v)));
            reg_term_triple_vec.push_back(Eigen::Triplet<float>(
                reg_term_cnt, v2_idx * 2 + 1, reg_term_w * (constant_v)));

            reg_term_triple_vec.push_back(Eigen::Triplet<float>(
                reg_term_cnt + 1, v1_idx * 2 + 1, reg_term_w * 1));
            reg_term_triple_vec.push_back(
                Eigen::Triplet<float>(reg_term_cnt + 1, v2_idx * 2 + 1,
                                      reg_term_w * (-1 + constant_u)));
            reg_term_triple_vec.push_back(Eigen::Triplet<float>(
                reg_term_cnt + 1, v3_idx * 2 + 1, reg_term_w * (-constant_u)));
            reg_term_triple_vec.push_back(Eigen::Triplet<float>(
                reg_term_cnt + 1, v3_idx * 2, reg_term_w * (constant_v)));
            reg_term_triple_vec.push_back(Eigen::Triplet<float>(
                reg_term_cnt + 1, v2_idx * 2, reg_term_w * (-constant_v)));

            reg_term_cnt += 2;
        }
        timer.TimeEnd();
        std::cout << "1 time: " << timer.TimeGap_in_ms() << std::endl;

        Eigen::SparseMatrix<float> reg_term_jacobian(
            reg_term_cnt, node_rows_ * node_cols_ * 2);
        reg_term_jacobian.setFromTriplets(reg_term_triple_vec.begin(),
                                          reg_term_triple_vec.end());
        Eigen::VectorXf reg_term_residual(reg_term_residual_vec.size());
        memcpy(reg_term_residual.data(), reg_term_residual_vec.data(),
               reg_term_residual_vec.size() * sizeof(float));

        /** 求解線性方程組JTJx=-JTb */
        Eigen::SparseMatrix<float> lm_term(node_rows_ * node_cols_ * 2,
                                           node_rows_ * node_cols_ * 2);
        std::vector<Eigen::Triplet<float> > lm_term_triple_vec;
        float lm_factor = 0.0000001f;
        for (int row = 0; row < lm_term.rows(); ++row) {
            lm_term_triple_vec.push_back(
                Eigen::Triplet<float>(row, row, lm_factor));
        }
        lm_term.setFromTriplets(lm_term_triple_vec.begin(),
                                lm_term_triple_vec.end());

        Eigen::SparseMatrix<float> JTJ =
            Eigen::SparseMatrix<float>(feat_corr_term_jacobian.transpose()) *
                feat_corr_term_jacobian +
            Eigen::SparseMatrix<float>(photo_term_jacobian.transpose()) *
                photo_term_jacobian +
            Eigen::SparseMatrix<float>(reg_term_jacobian.transpose()) *
                reg_term_jacobian +
            Eigen::SparseMatrix<float>(zero_term_jacobian.transpose()) *
                zero_term_jacobian;
        Eigen::VectorXf JTb =
            Eigen::SparseMatrix<float>(feat_corr_term_jacobian.transpose()) *
                feat_corr_term_residual +
            Eigen::SparseMatrix<float>(photo_term_jacobian.transpose()) *
                photo_term_residual +
            Eigen::SparseMatrix<float>(reg_term_jacobian.transpose()) *
                reg_term_residual +
            Eigen::SparseMatrix<float>(zero_term_jacobian.transpose()) *
                zero_term_residual;

        Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > chol(JTJ);
        Eigen::VectorXf delta_x = chol.solve(-JTb);

        for (int node_idx = 0; node_idx < node_vec_.size(); ++node_idx) {
            node_vec_[node_idx].x += delta_x(2 * node_idx);
            node_vec_[node_idx].y += delta_x(2 * node_idx + 1);
        }
    }

#if 1
    /** Show當前結果 */
    WarpSrcImg();
    ShowComparisonImg();
#endif
}
