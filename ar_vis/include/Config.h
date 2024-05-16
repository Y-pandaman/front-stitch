#ifndef CYLINDER_STITCHER_CONFIG_H
#define CYLINDER_STITCHER_CONFIG_H

#include <QDebug>
#include <QVector3D>
#include <opencv2/opencv.hpp>
#include <string>

class Config {
public:
    Config();
    bool load_config_file(const std::string& file_path);

    std::string blade_model_path;   // 铲刀模型保存路径
    std::string model_type;         // 铲刀模型文件类型, .obj or .ply

    QVector3D front_wheel_center;

    float track_min_show_v;

    QVector3D blade_color_0;
    QVector3D blade_color_1;

    float font_pixel_height;

    int back_track_width, back_track_height;

    int chandao_left_vertex_idx, chandao_right_vertex_idx;

    QString wheel_track_img_path, blade_track_img_path;

    QString track_label_dir;

    QString track_label_texture_path;

    float wheel_track_width, blade_track_width;

    float wheel_track_speed;

private:
    QVector3D convertToQVector3D(const cv::Mat& input);
};

extern Config config;

#endif   // CYLINDER_STITCHER_CONFIG_H
