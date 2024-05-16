#include "math_utils.h"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <vector>

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

void CalculateDataJTJ(cv::Mat src_img, cv::Mat target_img, cv::Mat dy_img,
                      cv::Mat dx_img, std::vector<float2> original_node_vec,
                      std::vector<float2> node_vec, float triangle_u,
                      float triangle_v, int img_rows, int img_cols,
                      int node_img_rows, int node_img_cols,
                      std::vector<int> pixel_rela_idx_vec,
                      std::vector<float> pixel_rela_weight_vec,
                      std::vector<int> node_rela_idx_vec,
                      std::vector<float> node_rela_weight_vec) {
    std::cout << "CalculateJTJ" << std::endl;

    std::vector<Eigen::Triplet<float> > photo_term_triple_vec;
    std::vector<float> photo_term_residual_vec;
    int photo_term_cnt = 0;
    int tl, tr, bl, br;
    for (int pixel_idx = 0; pixel_idx < pixel_rela_idx_vec.size() / 4;
         ++pixel_idx) {
        int row = pixel_idx / img_cols;
        int col = pixel_idx % img_cols;

        if (row < img_rows && col < img_cols) {
            tl = 4 * pixel_idx;
            tr = 4 * pixel_idx + 1;
            bl = 4 * pixel_idx + 2;
            br = 4 * pixel_idx + 3;

            float2 warped_pixel =
                pixel_rela_weight_vec[tl] * node_vec[pixel_rela_idx_vec[tl]] +
                pixel_rela_weight_vec[tr] * node_vec[pixel_rela_idx_vec[tr]] +
                pixel_rela_weight_vec[bl] * node_vec[pixel_rela_idx_vec[bl]] +
                pixel_rela_weight_vec[br] * node_vec[pixel_rela_idx_vec[br]];
            float2 src_image_uv    = warped_pixel;
            float2 target_image_uv = make_float2(col, row);
            float src_intensity =
                GetPixelValueBilinear(src_img, src_image_uv.x, src_image_uv.y);
            float dy_val =
                GetPixelValueBilinear(dy_img, src_image_uv.x, src_image_uv.y);
            float dx_val =
                GetPixelValueBilinear(dx_img, src_image_uv.x, src_image_uv.y);
            float target_intensity = GetPixelValueBilinear(
                target_img, target_image_uv.x, target_image_uv.y);

            if (target_intensity > 10.0f || src_intensity > 10.0f ||
                fabs(dy_val) > 10.0f || fabs(dx_val) > 10.0f) {
                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec[tl] * 2, 0.0));
                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec[tl] * 2 + 1, 0.0));

                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec[tr] * 2, 0.0));
                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec[tr] * 2 + 1, 0.0));

                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec[bl] * 2, 0.0));
                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec[bl] * 2 + 1, 0.0));

                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec[br] * 2, 0.0));
                photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                    photo_term_cnt, pixel_rela_idx_vec[br] * 2 + 1, 0.0));

                photo_term_residual_vec.push_back(0.0f);
                ++photo_term_cnt;
                continue;
            }

            float photo_term_w = 1.0f;
            photo_term_residual_vec.push_back(
                photo_term_w * (src_intensity - target_intensity));

            photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                photo_term_cnt, pixel_rela_idx_vec[tl] * 2,
                photo_term_w * dx_val * pixel_rela_weight_vec[tl]));
            photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                photo_term_cnt, pixel_rela_idx_vec[tl] * 2 + 1,
                photo_term_w * dy_val * pixel_rela_weight_vec[tl]));

            photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                photo_term_cnt, pixel_rela_idx_vec[tr] * 2,
                photo_term_w * dx_val * pixel_rela_weight_vec[tr]));
            photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                photo_term_cnt, pixel_rela_idx_vec[tr] * 2 + 1,
                photo_term_w * dy_val * pixel_rela_weight_vec[tr]));

            photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                photo_term_cnt, pixel_rela_idx_vec[bl] * 2,
                photo_term_w * dx_val * pixel_rela_weight_vec[bl]));
            photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                photo_term_cnt, pixel_rela_idx_vec[bl] * 2 + 1,
                photo_term_w * dy_val * pixel_rela_weight_vec[bl]));

            photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                photo_term_cnt, pixel_rela_idx_vec[br] * 2,
                photo_term_w * dx_val * pixel_rela_weight_vec[br]));
            photo_term_triple_vec.push_back(Eigen::Triplet<float>(
                photo_term_cnt, pixel_rela_idx_vec[br] * 2 + 1,
                photo_term_w * dy_val * pixel_rela_weight_vec[br]));
        }
        ++photo_term_cnt;
    }
    std::cout << std::endl;
    std::system("pause");

    Eigen::SparseMatrix<float> photo_term_jacobian(
        photo_term_cnt, node_img_rows * node_img_cols * 2);
    photo_term_jacobian.setFromTriplets(photo_term_triple_vec.begin(),
                                        photo_term_triple_vec.end());
    Eigen::VectorXf photo_term_residual(photo_term_residual_vec.size());
    memcpy(photo_term_residual.data(), photo_term_residual_vec.data(),
           photo_term_residual_vec.size() * sizeof(float));

    Eigen::SparseMatrix<float> JTJ =
        Eigen::SparseMatrix<float>(photo_term_jacobian.transpose()) *
        photo_term_jacobian;
    Eigen::VectorXf JTb =
        Eigen::SparseMatrix<float>(photo_term_jacobian.transpose()) *
        photo_term_residual;

    Eigen::MatrixXf dense_JTJ = JTJ.toDense();
    std::cout << "dense_JTJ(1200, 1200): " << dense_JTJ(1200, 1200)
              << std::endl;
    std::cout << "dense_JTJ(1200, 1201): " << dense_JTJ(1200, 1201)
              << std::endl;
    std::cout << "dense_JTJ(1201, 1200): " << dense_JTJ(1201, 1200)
              << std::endl;
    std::cout << "dense_JTJ(1201, 1201): " << dense_JTJ(1201, 1201)
              << std::endl;

    cv::Mat output_image(node_img_rows * node_img_cols * 2,
                         node_img_rows * node_img_cols * 2, CV_32FC1);
    std::cout << "--------------" << std::endl;
    for (int row = 0; row < node_img_rows * node_img_cols * 2; ++row) {
        for (int col = 0; col < node_img_rows * node_img_cols * 2; ++col) {
            output_image.at<float>(row, col) = dense_JTJ(row, col) * 100.0f;
        }
    }
    std::cout << "--------------" << std::endl;
    cv::resize(output_image, output_image,
               cv::Size(output_image.cols / 4, output_image.rows / 4));
    cv::imshow("output_image", output_image);
    cv::waitKey(1);
}

void CSRToMat(std::vector<int> ia_vec, std::vector<int> ja_vec,
              std::vector<float> a_vec, int node_num) {
    std::vector<Eigen::Triplet<float> > JTJ_triple_vec;
    int JTJ_rows = node_num * 2;
    for (int row = 0; row < JTJ_rows; ++row) {
        for (int idx = ia_vec[row]; idx < ia_vec[row + 1]; ++idx) {
            JTJ_triple_vec.push_back(
                Eigen::Triplet<float>(row, ja_vec[idx], a_vec[idx]));
        }
    }
    Eigen::SparseMatrix<float> JTJ(node_num * 2, node_num * 2);
    JTJ.setFromTriplets(JTJ_triple_vec.begin(), JTJ_triple_vec.end());

    Eigen::MatrixXf dense_JTJ = JTJ.toDense();
    std::cout << "dense_JTJ(1200, 1200): " << dense_JTJ(1200, 1200)
              << std::endl;
    std::cout << "dense_JTJ(1200, 1201): " << dense_JTJ(1200, 1201)
              << std::endl;
    std::cout << "dense_JTJ(1201, 1200): " << dense_JTJ(1201, 1200)
              << std::endl;
    std::cout << "dense_JTJ(1201, 1201): " << dense_JTJ(1201, 1201)
              << std::endl;

    cv::Mat output_image(node_num * 2, node_num * 2, CV_32FC1);
    for (int row = 0; row < node_num * 2; ++row) {
        for (int col = 0; col < node_num * 2; ++col) {
            output_image.at<float>(row, col) = dense_JTJ(row, col) * 100.0f;
        }
    }
    cv::resize(output_image, output_image,
               cv::Size(output_image.cols / 4, output_image.rows / 4));
    cv::imshow("output_image_2", output_image);
    cv::waitKey(0);

    std::system("pause");
    std::system("pause");
    std::system("pause");
}

void CalculateSmoothJTJ(cv::Mat src_img, cv::Mat target_img, cv::Mat dy_img,
                        cv::Mat dx_img, std::vector<float2> original_node_vec,
                        std::vector<float2> node_vec, float triangle_u,
                        float triangle_v, int img_rows, int img_cols,
                        int node_img_rows, int node_img_cols,
                        std::vector<int> pixel_rela_idx_vec,
                        std::vector<float> pixel_rela_weight_vec,
                        std::vector<int> node_rela_idx_vec,
                        std::vector<float> node_rela_weight_vec) {
    std::cout << "CalculateJTJ" << std::endl;

    float reg_term_w = 1.0f;

    /** Ӌ��reg_term��jacobian��residual��[xy,xy,xy,xy...] */
    std::vector<Eigen::Triplet<float> > reg_term_triple_vec;
    std::vector<float> reg_term_residual_vec;
    int reg_term_cnt = 0;
    for (int triangle_idx = 0; triangle_idx < node_rela_idx_vec.size() / 3;
         ++triangle_idx) {
        int v1_idx       = node_rela_idx_vec[3 * triangle_idx];
        int v2_idx       = node_rela_idx_vec[3 * triangle_idx + 1];
        int v3_idx       = node_rela_idx_vec[3 * triangle_idx + 2];
        float2 v1        = node_vec[v1_idx];
        float2 v2        = node_vec[v2_idx];
        float2 v3        = node_vec[v3_idx];
        float weight     = node_rela_weight_vec[triangle_idx];
        float constant_u = triangle_u;
        float constant_v = triangle_v;

        reg_term_residual_vec.push_back(
            reg_term_w * (v1.x - (v2.x + constant_u * (v3.x - v2.x) +
                                  constant_v * (v3.y - v2.y))));
        reg_term_residual_vec.push_back(
            reg_term_w * (v1.y - (v2.y + constant_u * (v3.y - v2.y) +
                                  constant_v * (-v3.x + v2.x))));

        reg_term_triple_vec.push_back(
            Eigen::Triplet<float>(reg_term_cnt, v1_idx * 2, reg_term_w * 1));
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
        reg_term_triple_vec.push_back(Eigen::Triplet<float>(
            reg_term_cnt + 1, v2_idx * 2 + 1, reg_term_w * (-1 + constant_u)));
        reg_term_triple_vec.push_back(Eigen::Triplet<float>(
            reg_term_cnt + 1, v3_idx * 2 + 1, reg_term_w * (-constant_u)));
        reg_term_triple_vec.push_back(Eigen::Triplet<float>(
            reg_term_cnt + 1, v3_idx * 2, reg_term_w * (constant_v)));
        reg_term_triple_vec.push_back(Eigen::Triplet<float>(
            reg_term_cnt + 1, v2_idx * 2, reg_term_w * (-constant_v)));

        reg_term_cnt += 2;
    }

    Eigen::SparseMatrix<float> photo_term_jacobian(
        reg_term_cnt, node_img_rows * node_img_cols * 2);
    photo_term_jacobian.setFromTriplets(reg_term_triple_vec.begin(),
                                        reg_term_triple_vec.end());
    Eigen::VectorXf photo_term_residual(reg_term_residual_vec.size());
    memcpy(photo_term_residual.data(), reg_term_residual_vec.data(),
           reg_term_residual_vec.size() * sizeof(float));

    Eigen::SparseMatrix<float> JTJ =
        Eigen::SparseMatrix<float>(photo_term_jacobian.transpose()) *
        photo_term_jacobian;
    Eigen::VectorXf JTb =
        Eigen::SparseMatrix<float>(photo_term_jacobian.transpose()) *
        photo_term_residual;

    std::cout << "JTb 1: " << std::endl;
    for (int i = 0; i < JTb.rows(); ++i) {
        std::cout << JTb(i, 0) << ", ";
    }
    std::cout << std::endl;

    Eigen::MatrixXf dense_JTJ = JTJ.toDense();
    std::cout << "dense_JTJ(1200, 1200): " << dense_JTJ(1200, 1200)
              << std::endl;
    std::cout << "dense_JTJ(1200, 1201): " << dense_JTJ(1200, 1201)
              << std::endl;
    std::cout << "dense_JTJ(1201, 1200): " << dense_JTJ(1201, 1200)
              << std::endl;
    std::cout << "dense_JTJ(1201, 1201): " << dense_JTJ(1201, 1201)
              << std::endl;

    cv::Mat output_image(node_img_rows * node_img_cols * 2,
                         node_img_rows * node_img_cols * 2, CV_32FC1);
    for (int row = 0; row < node_img_rows * node_img_cols * 2; ++row) {
        for (int col = 0; col < node_img_rows * node_img_cols * 2; ++col) {
            output_image.at<float>(row, col) = dense_JTJ(row, col) * 100.0f;
        }
    }
    cv::resize(output_image, output_image,
               cv::Size(output_image.cols / 4, output_image.rows / 4));
    cv::imshow("output_image", output_image);
    cv::waitKey(0);

    std::system("pause");
}

void CalculateZeroJTJ(cv::Mat src_img, cv::Mat target_img, cv::Mat dy_img,
                      cv::Mat dx_img, std::vector<float2> original_node_vec,
                      std::vector<float2> node_vec, float triangle_u,
                      float triangle_v, int img_rows, int img_cols,
                      int node_img_rows, int node_img_cols,
                      std::vector<int> pixel_rela_idx_vec,
                      std::vector<float> pixel_rela_weight_vec,
                      std::vector<int> node_rela_idx_vec,
                      std::vector<float> node_rela_weight_vec) {
    std::cout << "CalculateJTJ" << std::endl;
    std::system("pause");

    float zero_term_w = 0.1f;

    std::vector<Eigen::Triplet<float> > zero_term_triple_vec;
    std::vector<float> zero_term_residual_vec;
    int zero_term_cnt = 0;
    for (int node_idx = 0; node_idx < node_vec.size(); ++node_idx) {
        zero_term_residual_vec.push_back(
            zero_term_w *
            (node_vec[node_idx].x - original_node_vec[node_idx].x));
        zero_term_residual_vec.push_back(
            zero_term_w *
            (node_vec[node_idx].y - original_node_vec[node_idx].y));

        zero_term_triple_vec.push_back(Eigen::Triplet<float>(
            zero_term_cnt, node_idx * 2, zero_term_w * 1));
        zero_term_triple_vec.push_back(Eigen::Triplet<float>(
            zero_term_cnt + 1, node_idx * 2 + 1, zero_term_w * 1));

        zero_term_cnt += 2;
    }
    Eigen::SparseMatrix<float> zero_term_jacobian(
        zero_term_cnt, node_img_rows * node_img_cols * 2);
    zero_term_jacobian.setFromTriplets(zero_term_triple_vec.begin(),
                                       zero_term_triple_vec.end());
    Eigen::VectorXf zero_term_residual(zero_term_residual_vec.size());
    memcpy(zero_term_residual.data(), zero_term_residual_vec.data(),
           zero_term_residual_vec.size() * sizeof(float));

    Eigen::SparseMatrix<float> JTJ =
        Eigen::SparseMatrix<float>(zero_term_jacobian.transpose()) *
        zero_term_jacobian;
    Eigen::VectorXf JTb =
        Eigen::SparseMatrix<float>(zero_term_jacobian.transpose()) *
        zero_term_residual;

    std::cout << "JTb 1: " << std::endl;
    for (int i = 0; i < JTb.rows(); ++i) {
        std::cout << JTb(i, 0) << ", ";
    }
    std::cout << std::endl;

    Eigen::MatrixXf dense_JTJ = JTJ.toDense();
    std::cout << "dense_JTJ(1200, 1200): " << dense_JTJ(1200, 1200)
              << std::endl;
    std::cout << "dense_JTJ(1200, 1201): " << dense_JTJ(1200, 1201)
              << std::endl;
    std::cout << "dense_JTJ(1201, 1200): " << dense_JTJ(1201, 1200)
              << std::endl;
    std::cout << "dense_JTJ(1201, 1201): " << dense_JTJ(1201, 1201)
              << std::endl;

    cv::Mat output_image(node_img_rows * node_img_cols * 2,
                         node_img_rows * node_img_cols * 2, CV_32FC1);
    for (int row = 0; row < node_img_rows * node_img_cols * 2; ++row) {
        for (int col = 0; col < node_img_rows * node_img_cols * 2; ++col) {
            output_image.at<float>(row, col) = dense_JTJ(row, col) * 100.0f;
        }
    }
    cv::resize(output_image, output_image,
               cv::Size(output_image.cols / 4, output_image.rows / 4));
    cv::imshow("output_image", output_image);
    cv::waitKey(0);
}

void CalculateDeltaX(std::vector<float>& out_delta_x, std::vector<int> ia_vec,
                     std::vector<int> ja_vec, std::vector<float> a_vec,
                     std::vector<float> jtb_vec, int node_num) {
    std::vector<Eigen::Triplet<float> > JTJ_triple_vec;
    int JTJ_rows = node_num * 2;
    for (int row = 0; row < JTJ_rows; ++row) {
        for (int idx = ia_vec[row]; idx < ia_vec[row + 1]; ++idx) {
            JTJ_triple_vec.push_back(
                Eigen::Triplet<float>(row, ja_vec[idx], a_vec[idx]));
        }
    }
    Eigen::SparseMatrix<float> JTJ(node_num * 2, node_num * 2);
    JTJ.setFromTriplets(JTJ_triple_vec.begin(), JTJ_triple_vec.end());

    Eigen::SparseMatrix<float> lm_term(node_num * 2, node_num * 2);
    std::vector<Eigen::Triplet<float> > lm_term_triple_vec;
    float lm_factor = 0.0000001f;
    for (int row = 0; row < lm_term.rows(); ++row) {
        lm_term_triple_vec.push_back(
            Eigen::Triplet<float>(row, row, lm_factor));
    }
    lm_term.setFromTriplets(lm_term_triple_vec.begin(),
                            lm_term_triple_vec.end());

    Eigen::VectorXf JTb(jtb_vec.size());
    memcpy(JTb.data(), jtb_vec.data(), jtb_vec.size() * sizeof(float));

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > chol(JTJ);
    Eigen::VectorXf delta_x = chol.solve(JTb);

    for (int node_idx = 0; node_idx < node_num; ++node_idx) {
        out_delta_x[2 * node_idx] += delta_x(2 * node_idx);
        out_delta_x[2 * node_idx + 1] += delta_x(2 * node_idx + 1);
    }
}
