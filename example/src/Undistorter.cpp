/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-09 18:53:19
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "Undistorter.h"

bool Undistorter::loadCameraIntrin(const std::string& fs_path) {
    cv::FileStorage fs;
    if (!fs.open(fs_path, cv::FileStorage::Mode::READ)) {
        printf("cannot open fs file %s\n", fs_path.c_str());
        return false;
    }
    fs["K"] >> K;
    fs["D"] >> D;
    fs["image_size"] >> input_image_size;

    getMapForRemapping(1.0, 0.0);
    map_inited = true;
    return true;
}

bool Undistorter::getMapForRemapping(float new_size_factor, float balance) {
    if (K.empty() || D.empty()) {
        printf("K & D empty, cannot get map for remapping\n");
        return false;
    }
    cv::Mat eye_mat = cv::Mat::eye(3, 3, CV_32F);
    this->new_image_size =
        cv::Size(this->input_image_size.width * new_size_factor,
                 this->input_image_size.height * new_size_factor);
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        K, D, input_image_size, eye_mat, new_K, balance, new_image_size);
    cv::fisheye::initUndistortRectifyMap(K, D, eye_mat, new_K, new_image_size,
                                         CV_16SC2, this->map1, this->map2);
    new_D = D;
    return true;
}

bool Undistorter::undistortImage(cv::Mat input_image, cv::Mat& output_image) {
    if (!map_inited) {
        bool flag = getMapForRemapping();
        if (!flag)
            return false;
    }
    cv::remap(input_image, output_image, map1, map2, cv::INTER_LINEAR);
    return true;
}

bool Undistorter::getMask(cv::Mat& out_mask) {
    if (!map_inited)
        return false;
    if (!mask_generated) {
        mask = cv::Mat::ones(input_image_size, CV_8UC1) * 255;
        cv::remap(mask, mask, map1, map2, cv::INTER_LINEAR);
        mask_generated = true;
    }

    out_mask = mask.clone();
    return true;
}
void Undistorter::changeIntrin(float factor) {
    K *= factor;
    K.at<double>(2, 2) = 1;
    input_image_size.width *= factor;
    input_image_size.height *= factor;
}
