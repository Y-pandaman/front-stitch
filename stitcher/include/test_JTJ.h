#pragma once

#include "math_utils.h"
#include <opencv2/opencv.hpp>

void CalculateDataJTJ(cv::Mat src_img, cv::Mat target_img, cv::Mat dy_img,
                      cv::Mat dx_img, std::vector<float2> original_node_vec,
                      std::vector<float2> node_vec, float triangle_u,
                      float triangle_v, int img_rows, int img_cols,
                      int node_img_rows, int node_img_cols,
                      std::vector<int> pixel_rela_idx_vec,
                      std::vector<float> pixel_rela_weight_vec,
                      std::vector<int> node_rela_idx_vec,
                      std::vector<float> node_rela_weight_vec);

void CalculateSmoothJTJ(cv::Mat src_img, cv::Mat target_img, cv::Mat dy_img,
                        cv::Mat dx_img, std::vector<float2> original_node_vec,
                        std::vector<float2> node_vec, float triangle_u,
                        float triangle_v, int img_rows, int img_cols,
                        int node_img_rows, int node_img_cols,
                        std::vector<int> pixel_rela_idx_vec,
                        std::vector<float> pixel_rela_weight_vec,
                        std::vector<int> node_rela_idx_vec,
                        std::vector<float> node_rela_weight_vec);

void CalculateZeroJTJ(cv::Mat src_img, cv::Mat target_img, cv::Mat dy_img,
                      cv::Mat dx_img, std::vector<float2> original_node_vec,
                      std::vector<float2> node_vec, float triangle_u,
                      float triangle_v, int img_rows, int img_cols,
                      int node_img_rows, int node_img_cols,
                      std::vector<int> pixel_rela_idx_vec,
                      std::vector<float> pixel_rela_weight_vec,
                      std::vector<int> node_rela_idx_vec,
                      std::vector<float> node_rela_weight_vec);

void CSRToMat(std::vector<int> ia_vec, std::vector<int> ja_vec,
              std::vector<float> a_vec, int node_num);

void CalculateDeltaX(std::vector<float>& out_delta_x, std::vector<int> ia_vec,
                     std::vector<int> ja_vec, std::vector<float> a_vec,
                     std::vector<float> jtb_vec, int node_num);
