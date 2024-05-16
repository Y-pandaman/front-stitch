/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-09 19:14:52
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "cylinder_stitcher.h"
#include "cylinder_stitcher.cuh"
#include <Eigen/Eigen>
#include <fstream>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

void CylinderStitcherGPU::setYawPitchRoll(
    std::vector<std::vector<int> > yawpitchroll) {
    for (int camera_idx = 0; camera_idx < view_num_; ++camera_idx) {
        std::cout << yawpitchroll[camera_idx][0] << ", "
                  << yawpitchroll[camera_idx][1] << ", "
                  << yawpitchroll[camera_idx][2] << std::endl;
        Eigen::Vector3f eulerAngle(
            yawpitchroll[camera_idx][0] * 3.141592653 / 180.0,
            yawpitchroll[camera_idx][1] * 3.141592653 / 180.0,
            yawpitchroll[camera_idx][2] * 3.141592653 / 180.0);
        Eigen::AngleAxisf yawAngle(
            Eigen::AngleAxisf(eulerAngle(0), Eigen::Vector3f::UnitY()));
        Eigen::AngleAxisf pitchAngle(
            Eigen::AngleAxisf(eulerAngle(1), Eigen::Vector3f::UnitX()));
        Eigen::AngleAxisf rollAngle(
            Eigen::AngleAxisf(eulerAngle(2), Eigen::Vector3f::UnitZ()));
        Eigen::AngleAxisf rotation_vector;
        rotation_vector                 = yawAngle * pitchAngle * rollAngle;
        Eigen::Matrix4f camera_pose     = Eigen::Matrix4f::Identity();
        camera_pose.topLeftCorner(3, 3) = rotation_vector.toRotationMatrix();
        Eigen::Matrix4f extrin          = camera_pose.inverse();
        std::vector<float> R, T, C;
        float *d_R, *d_T, *d_C;
        R.emplace_back(extrin(0, 0));
        R.emplace_back(extrin(0, 1));
        R.emplace_back(extrin(0, 2));
        R.emplace_back(extrin(1, 0));
        R.emplace_back(extrin(1, 1));
        R.emplace_back(extrin(1, 2));
        R.emplace_back(extrin(2, 0));
        R.emplace_back(extrin(2, 1));
        R.emplace_back(extrin(2, 2));
        checkCudaErrors(cudaMalloc((void**)&d_R, 9 * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_R, R.data(), 9 * sizeof(float),
                                   cudaMemcpyHostToDevice));

        T.emplace_back(extrin(0, 3));
        T.emplace_back(extrin(1, 3));
        T.emplace_back(extrin(2, 3));
        checkCudaErrors(cudaMalloc((void**)&d_T, 3 * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_T, T.data(), 3 * sizeof(float),
                                   cudaMemcpyHostToDevice));

        C.emplace_back(camera_pose(0, 3));
        C.emplace_back(camera_pose(1, 3));
        C.emplace_back(camera_pose(2, 3));
        checkCudaErrors(cudaMalloc((void**)&d_C, 3 * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_C, C.data(), 3 * sizeof(float),
                                   cudaMemcpyHostToDevice));

        views_[camera_idx].camera.free();
        views_[camera_idx].camera.R = d_R;
        views_[camera_idx].camera.T = d_T;
        views_[camera_idx].camera.C = d_C;
    }
}
