/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-15 17:02:45
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#pragma once

#include "math_utils.h"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <string>

class ImageAlignmentCPU {
public:
    ImageAlignmentCPU(int img_rows, int img_cols, int grid_img_rows,
                      int grid_img_cols);
    ~ImageAlignmentCPU();

    void SetSrcTargetImgs(cv::Mat& src_img, cv::Mat& src_mask, int2 src_min_uv,
                          cv::Mat& target_img, cv::Mat& target_mask,
                          int2 target_min_uv, std::vector<float4> match_corr);
    void AlignSrcImgToTarget(int sample_step = 1);
    void ShowNodeImg(std::vector<std::vector<float2> >& node_vec_vis);
    void ShowComparisonImg();
    void WarpSrcImg();
    void WarpSrcImg(cv::Mat_<cv::Vec3f>& target_image,
                    cv::Mat_<cv::Vec3f>& src_image);

public:
    // 定义与匹配相关的变量
    std::vector<float4> match_corr_;   // 存储匹配相关的向量

    // 定义图像变量
    cv::Mat src_img_, blur_src_img_;         // 源图像及其模糊版本
    cv::Mat warped_src_img_;                 // 变形后的源图像
    cv::Mat target_img_, blur_target_img_;   // 目标图像及其模糊版本
    cv::Mat_<float> dy_img_, dx_img_;   // 目标图像的梯度图像（y方向和x方向）

    // 定义图像坐标变量
    float2 src_min_uv_, target_min_uv_;   // 源图像和目标图像的最小UV坐标

    // 定义图像尺寸和其他相关尺寸
    int img_rows_, img_cols_, grid_img_rows_, grid_img_cols_, node_rows_,
        node_cols_, grid_width_,
        grid_height_;   // 图像行数、列数，网格图像行数、列数，节点行数、列数，网格宽度和高度
    // 存储节点的向量，每个节点包含两个float类型的坐标值
    std::vector<float2> node_vec_;
    // 存储像素与节点关系的索引向量，每个像素对应一个节点的索引
    std::vector<int> pixel_rela_idx_vec_;
    // 存储像素与节点关系的权重向量，对应于pixel_rela_idx_vec_中每个索引的权重值
    std::vector<float> pixel_rela_weight_vec_;
    // 存储节点之间关系的索引向量，每个节点关系对应一个节点对的索引
    std::vector<int> node_rela_idx_vec_;
    // 存储节点之间关系的权重向量，对应于node_rela_idx_vec_中每个索引对的权重值
    std::vector<float> node_rela_weight_vec_;
    // 存储节点的u坐标向量
    std::vector<float> node_u_vec_;
    // 存储节点的v坐标向量
    std::vector<float> node_v_vec_;

    // 存储特征相关项的系数和残差
    std::vector<Eigen::Triplet<float> > feat_corr_term_triple_vec_;
    std::vector<float> feat_corr_term_residual_vec_;

    // 存储照片项的系数和残差
    std::vector<Eigen::Triplet<float> > photo_term_triple_vec_;
    std::vector<float> photo_term_residual_vec_;

    // 存储零项的系数和残差，零项通常用于表示不参与优化的项
    std::vector<Eigen::Triplet<float> > zero_term_triple_vec_;
    std::vector<float> zero_term_residual_vec_;

    // 存储正则化项的系数和残差，用于增加优化过程的稳定性或引入额外的约束
    std::vector<Eigen::Triplet<float> > reg_term_triple_vec_;
    std::vector<float> reg_term_residual_vec_;
};
