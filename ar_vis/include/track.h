/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-16 20:15:21
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
//
// Created by touch on 22-11-1.
//

#ifndef DRAW_TRACK_TRACK_H
#define DRAW_TRACK_TRACK_H

#include "TextSourceManager.h"
#include <Eigen/Eigen>
#include <QPoint>
#include <iostream>

#define TRACK_TEXTURE_COORD_LEN 2
#define TRACK_POS_COORD_LEN 3
#define TRACK_POINT_LEN (TRACK_TEXTURE_COORD_LEN + TRACK_POS_COORD_LEN)

class Track {
private:
    float track_radius, track_width, track_length;   // m
    float mid_radius;   // for distance, which is same as another track
    float begin_angle;
    Eigen::Vector3f center;   // center pos,
    float* point_list;
    std::vector<float> label_point_list;
    int orientation = 1;   // -1 for turn left, 1 for turn right
    int track_loc;         // -1 left track, 1 right track
    int segment_num, point_len;
    int point_num;
    bool up_to_date          = false;
    bool have_cross          = false;
    bool have_number_label   = true;
    float number_label_scale = 0.015;
    bool point_num_changed   = false;
    int cross_num;
    float cross_loc = 0;
    float cross_width, cross_interval, cross_len;
    float interval_factor = 1;

    const TextSourceManager* text_source_manager;

    void generatePointList();

public:
    explicit Track(int _loc, float _radius = 1000, float _mid_radius = 1000,
                   float _width = 0.1, float _length = 40,
                   Eigen::Vector3f _center = Eigen::Vector3f(1000, 0, 0),
                   int _segment_num        = 1000);

    void setTrack(int _loc, float _radius = 1000, float _mid_radius = 1000,
                  float _width = 0.1, float _length = 40,
                  const Eigen::Vector3f& _center = Eigen::Vector3f(1000, 0, 0),
                  int _segment_num               = 1000);

    ~Track();

    void set(float _radius, float _mid_radius, int _ori,
             const Eigen::Vector3f& _center, float _begin_angle, float _width);

    float* getPointList();

    /**
     * 获取点的数量
     *
     * 本函数不接受任何参数。
     *
     * @return 返回当前点的数量。
     */
    int getPointNum() const {
        return point_num;   // 返回点的数量
    }

    const std::vector<float>& getLabelPointList() {
        return label_point_list;
    }

    uint getBufferSize() const;

    void print_point_list(std::string filename);

    // angle hudu
    // interval m
    void setCross(float width, float loc, float interval, float len,
                  float _interval_factor);

    void setTextSourceManager(const TextSourceManager* _text_source_manager) {
        text_source_manager = _text_source_manager;
    }
};

#endif   // DRAW_TRACK_TRACK_H
