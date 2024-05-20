/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-09 18:21:56
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
//
// Created by touch on 22-10-17.
//

#ifndef CALIB_GROUND_TAGPOSESSET_H
#define CALIB_GROUND_TAGPOSESSET_H

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <vector>

class TagPosesSet {
public:
    int poses_num = 0;
    std::vector<cv::Mat> pose_matrix_list;
    std::vector<int> tags_id;

    bool readFromFileStorage(const cv::FileStorage& fs) {
        fs["poses_num"] >> poses_num;
        for (int x = 0; x < poses_num; x++) {
            if (x >= poses_num)
                break;
            cv::Mat pose_matrix;
            int id;
            fs["pose_" + std::to_string(x)] >> pose_matrix;
            fs["tag_" + std::to_string(x)] >> id;
            pose_matrix_list.push_back(pose_matrix);
            tags_id.push_back(id);
        }
        return true;
    }
};

#endif   // CALIB_GROUND_TAGPOSESSET_H
