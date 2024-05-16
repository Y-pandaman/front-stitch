#include "Config.h"

bool Config::load_config_file(const std::string& file_path) {
    cv::FileStorage fs;
    if (!fs.open(file_path, cv::FileStorage::READ)) {
        printf("cannot open config file %s\n", file_path.c_str());
        exit(0);
    }
    fs["model_type"] >> model_type;
    fs["blade_model_path"] >> blade_model_path;

    cv::Mat temp;
    fs["front_wheel_center"] >> temp;
    front_wheel_center = convertToQVector3D(temp);

    fs["track_min_show_v"] >> track_min_show_v;

    fs["blade_color_0"] >> temp;
    blade_color_0 = convertToQVector3D(temp);

    fs["blade_color_1"] >> temp;
    blade_color_1 = convertToQVector3D(temp);

    fs["font_pixel_height"] >> font_pixel_height;

    fs["back_track_width"] >> back_track_width;
    fs["back_track_height"] >> back_track_height;

    fs["chandao_left_vertex_idx"] >> chandao_left_vertex_idx;
    fs["chandao_right_vertex_idx"] >> chandao_right_vertex_idx;

    std::string tmp;
    fs["wheel_track_img_path"] >> tmp;
    wheel_track_img_path = QString::fromStdString(tmp);
    fs["blade_track_img_path"] >> tmp;
    blade_track_img_path = QString::fromStdString(tmp);

    fs["track_label_dir"] >> tmp;
    track_label_dir = QString::fromStdString(tmp);

    fs["track_label_texture_path"] >> tmp;
    track_label_texture_path = QString::fromStdString(tmp);

    fs["wheel_track_width"] >> wheel_track_width;
    fs["blade_track_width"] >> blade_track_width;

    fs["wheel_track_speed"] >> wheel_track_speed;
    return true;
}

QVector3D Config::convertToQVector3D(const cv::Mat& input) {
    return {input.at<float>(0, 0), input.at<float>(1, 0),
            input.at<float>(2, 0)};
}

Config::Config() {
    load_config_file("../example/yamls/config.yaml");
}

Config config;