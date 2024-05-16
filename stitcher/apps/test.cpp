#include "include/cylinder_stitcher.cuh"
#include "include/cylinder_stitcher.h"
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/rgbd/large_kinfu.hpp>
#include <stack>

void LoadCameras(std::vector<PinholeCamera>& cameras, std::string path) {
    cameras.clear();

    std::ifstream input_file;
    input_file.open(path);
    std::cout << path << std::endl;

    std::string line;

    int view_idx = 0;
    int width, height;
    float4 intrin;
    float near_depth, far_depth, fx, fy, cx, cy;
    Eigen::Matrix4f camera_pose;
    bool use_realitycapture = true;
    int count               = 0, init_width, init_height;
    while (std::getline(input_file, line)) {
        std::cout << line << std::endl;

        sscanf(line.c_str(), "%d", &view_idx);

        std::getline(input_file, line);
        sscanf(line.c_str(), "%f %f %f %f", &fx, &fy, &cx, &cy);
        std::getline(input_file, line);
        sscanf(line.c_str(), "%d %d %f %f", &width, &height, &near_depth,
               &far_depth);
        for (int i = 0; i < 4; ++i) {
            std::getline(input_file, line);
            sscanf(line.c_str(), "%f %f %f %f", &camera_pose(i, 0),
                   &camera_pose(i, 1), &camera_pose(i, 2), &camera_pose(i, 3));
        }
        std::cout << camera_pose << std::endl;
        if (count == 0) {
            init_width  = width;
            init_height = height;
        }

        Eigen::Matrix3f K;
        K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        Eigen::Matrix3f R = camera_pose.topLeftCorner(3, 3);
        Eigen::Vector3f T = camera_pose.topRightCorner(3, 1);
        cameras.emplace_back(PinholeCamera(K, R, T, true));

        ++count;
    }

    input_file.close();
}

int main(int argc, char** argv) {
    Eigen::Matrix4f extrin_1, extrin_2, extrin_3;
    extrin_1 << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    extrin_2 << 0.7861, -0.0709, 0.6140, 59.5620, 0.0453, 0.9973, 0.0571,
        3.0746, -0.6164, -0.0170, 0.7872, -15.8743, 0, 0, 0, 1.0000;
    extrin_3 << 0.2341, -0.0632, 0.9701, 92.8090, 0.0436, 0.9976, 0.0545,
        3.0016, -0.9712, 0.0295, 0.2363, -69.2005, 0, 0, 0, 1.0000;
    extrin_1 = extrin_1.inverse();
    extrin_2 = extrin_2.inverse();
    extrin_3 = extrin_3.inverse();
    std::cout << extrin_1.topLeftCorner(3, 3).transpose() *
                     extrin_1.topLeftCorner(3, 3)
              << std::endl;
    std::cout << extrin_2.topLeftCorner(3, 3).transpose() *
                     extrin_2.topLeftCorner(3, 3)
              << std::endl;
    std::cout << extrin_3.topLeftCorner(3, 3).transpose() *
                     extrin_3.topLeftCorner(3, 3)
              << std::endl;
    extrin_1.topRightCorner(3, 1) = extrin_2.topRightCorner(3, 1);
    extrin_2.topRightCorner(3, 1) = extrin_2.topRightCorner(3, 1);
    extrin_3.topRightCorner(3, 1) = extrin_2.topRightCorner(3, 1);
    extrin_1                      = extrin_1.inverse();
    extrin_2                      = extrin_2.inverse();
    extrin_3                      = extrin_3.inverse();
    std::vector<PinholeCamera> cameras;

    int image_num = 3;
    std::vector<View> views;
    views.resize(image_num);
    std::vector<std::vector<float4> > intrins_vec;
    std::vector<std::vector<float> > extrins_vec;
    std::vector<std::vector<uchar3> > images_vec;
    std::vector<std::vector<uchar> > masks_vec;
    std::string data_dir;

    bool is_first_frame = true;
    int view_idx_map[]  = {0, 1, 2};
    for (int image_idx = 500; image_idx < 700; ++image_idx) {
        printf("%d\n", image_idx);
        std::vector<float4> intrins;
        std::vector<float> extrins;
        std::vector<uchar3> images;
        std::vector<uchar> masks;

        for (int view_count = 0; view_count < image_num; ++view_count) {
            int view_idx = view_idx_map[view_count];

            char name[512];
            sprintf(name, (data_dir + "%05d_color_%d.png").c_str(), image_idx,
                    view_idx + 1);
            cv::Mat image = cv::imread(name);
            sprintf(
                name,
                (data_dir + "../" + std::to_string(view_idx + 1) + "_mask.png")
                    .c_str(),
                view_idx + 1);
            cv::Mat mask_image = cv::imread(name, 0);
            cv::Mat erode_kernel =
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::erode(mask_image, mask_image, erode_kernel);

            Eigen::Matrix3f K;
            Eigen::Matrix4f temp;
            if (view_count == 0) {
                K << 3.2162245782458695e+02, 0., 6.8041318276762649e+02, 0.,
                    3.2180460006502807e+02, 3.1714482022084769e+02, 0., 0., 1.;
                intrins.emplace_back(
                    make_float4(K(0, 0), K(1, 1), K(0, 2), K(1, 2)));
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        extrins.emplace_back(extrin_1(row, col));
                    }
                }
            }
            if (view_count == 1) {
                K << 3.2419858812128535e+02, 0., 6.8578723003349660e+02, 0.,
                    3.2441589334815450e+02, 3.4752893367104724e+02, 0., 0., 1.;
                intrins.emplace_back(
                    make_float4(K(0, 0), K(1, 1), K(0, 2), K(1, 2)));
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        extrins.emplace_back(extrin_2(row, col));
                    }
                }
            }
            if (view_count == 2) {
                K << 3.2417781618030443e+02, 0., 6.7165929616664471e+02, 0.,
                    3.2413804460808342e+02, 3.1820741716475891e+02, 0., 0., 1.;
                intrins.emplace_back(
                    make_float4(K(0, 0), K(1, 1), K(0, 2), K(1, 2)));
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        extrins.emplace_back(extrin_3(row, col));
                    }
                }
            }
            PinholeCamera camera(K, temp.topLeftCorner(3, 3),
                                 temp.topRightCorner(3, 1), false);

            views[view_count].camera = camera;
            views[view_count].image  = image.clone();

            if (is_first_frame) {
                views[view_count].mask = mask_image.clone();
            }

            for (int row = 0; row < image.rows; ++row) {
                for (int col = 0; col < image.cols; ++col) {
                    images.emplace_back(image.at<uchar3>(row, col));
                    masks.emplace_back(mask_image.at<uchar>(row, col));
                }
            }
        }
        intrins_vec.emplace_back(intrins);
        extrins_vec.emplace_back(extrins);
        images_vec.emplace_back(images);
        masks_vec.emplace_back(masks);

        is_first_frame = false;
    }

    int width = views[0].image.cols, height = views[0].image.rows;
    ;

    CylinderStitcherGPU cylinder_stitcher(image_num, 2000 * 1000 * image_num);
    cylinder_stitcher.setMasks(masks_vec[0].data(), width, height, image_num);
    float3 origin =
        make_float3(views[1].camera.C(0, 0), views[1].camera.C(1, 0),
                    views[1].camera.C(2, 0));
    origin = origin - 40 * make_float3(views[1].camera.R(2, 0),
                                       views[1].camera.R(2, 1),
                                       views[1].camera.R(2, 2));
    cylinder_stitcher.novel_view_pos_ = origin;
    std::cout << "origin " << origin.x << " " << origin.y << " " << origin.z
              << std::endl;

    cv::Mat image, mask;
    cv::VideoWriter* video_writer = nullptr;
    for (int image_idx = 0; image_idx < intrins_vec.size(); ++image_idx) {
        cylinder_stitcher.setImages(images_vec[image_idx].data(), width, height,
                                    image_num);
        cylinder_stitcher.getCylinderImageCPU(image, mask);
        cv::imshow("image", image);
        cv::waitKey(1);
        if (video_writer == nullptr) {
            video_writer = new cv::VideoWriter;
            video_writer->open("H:/render_video.mp4",
                               cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 25,
                               cv::Size(image.cols, image.rows), true);
        }
        (*video_writer) << image;
    }
    video_writer->release();

    return 0;
}