/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-15 14:06:12
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#ifndef DRIVEASSISTANT_H
#define DRIVEASSISTANT_H

#include "BladeModelTransformer.h"
#include "Config.h"
#include "Model.h"
#include "track.h"
#include <QDebug>
#include <QVector2D>
#include <cmath>

extern Config config;

class DriveAssistant {
public:
    DriveAssistant(float wheel_w_dis_, float wheel_h_dis_,
                   float cross_interval_, float track_length_,
                   float track_width_, int segment_num_);

    ~DriveAssistant();

    void setDriveAssistant(float wheel_w_dis_, float wheel_h_dis_,
                           float cross_interval_, float track_length_,
                           float track_width_);

    void adjustLeftWheelDeltaTheta(float delta_theta);

    void adjustLeftWheelTheta(float theta);

    void setTrackCross(float width, float loc, float interval, float len);

    void update();

    void saveTrackDelta();

public:
    float left_wheel_theta = 0.0, right_wheel_theta = 0.0;
    float wheel_w_dis;      // m
    float wheel_h_dis;      // m
    float cross_interval;   // m
    float track_length;     // m
    float track_width;      // m

    float* cross_point_list = nullptr;
    float left_radius, right_radius;

    float track_delta_x = 0, track_delta_y = 0, track_delta_interval_factor = 1;
    void adjustTrackDelta(float delta_x, float delta_y, float delta_factor);

    QVector2D circle_origin;

    int turning_direction;   //  -1 for left, 1 for right
    Track left_track, right_track, left_safe_track, left_blade_track,
        right_safe_track, right_blade_track;
    Track back_left_track, back_right_track;

    void init_blade_model_transformer(const QVector3D& _p_A,
                                      const QVector3D& _p_D,
                                      const QVector3D& _p_F,
                                      const QVector3D& _p_G, float _l_AE,
                                      float _l_AB, float _l_BC, float _l_AC,
                                      const QVector3D& _p_2_nC,
                                      const QVector3D& _p_2_nA, float _l_nBnC);

    QMatrix4x4 blade_view_matrix;
    BladeModelTransformer* blade_model_transformer = nullptr;
    QVector3D front_wheel_center, left_max_blade_point, right_max_blade_point;

private:
    std::filesystem::path track_delta_yaml_path;
    /**
     * 计算从给定点到地面轴线的向量
     *
     * @param point
     * 输入参数，表示待计算的点的坐标，是一个Eigen::Vector3f类型向量
     * @return 返回从给定点到地面轴线的向量，也是一个Eigen::Vector3f类型向量
     */
    Eigen::Vector3f getBladeAxisToGroundAxis(const Eigen::Vector3f& point) {
        // 计算从front_wheel_center到给定点的向量，然后减去0.6单位长度，保持在同一水平面上
        Eigen::Vector3f res =
            Eigen::Vector3f(front_wheel_center.x() - point.x() - 0.6,
                            front_wheel_center.y() - point.y(), 0);
        return res;
    }
};

#endif   // DRIVEASSISTANT_H
