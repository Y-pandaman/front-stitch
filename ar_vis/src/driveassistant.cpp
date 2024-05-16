#include "driveassistant.h"
#include <iostream>

DriveAssistant::DriveAssistant(float wheel_w_dis_, float wheel_h_dis_,
                               float cross_interval_, float track_length_,
                               float track_width_, int segment_num_)
    : wheel_w_dis(wheel_w_dis_), wheel_h_dis(wheel_h_dis_),
      cross_interval(cross_interval_), track_length(track_length_),
      track_width(track_width_),
      left_track(-1, 1000, 1000, 0.2, 40, Eigen::Vector3f(1000, 0, 0), 1000),
      right_track(1, 1000, 1000, 0.2, 40, Eigen::Vector3f(1000, 0, 0), 1000),
      left_safe_track(-1), left_blade_track(-1), right_safe_track(1),
      right_blade_track(1), back_left_track(-1), back_right_track(1)

{
    left_radius = right_radius = 1000;
    turning_direction          = 1;
    cv::FileStorage fs;
    track_delta_yaml_path =
        std::filesystem::path("../example/yamls/track_delta.yaml");
    if (!fs.open(track_delta_yaml_path, cv::FileStorage::READ)) {
        printf("cannot open %s\n", track_delta_yaml_path.c_str());
    } else {
        //        fs["track_delta_x"] >> track_delta_x;
        //        fs["track_delta_y"] >> track_delta_y;
        //        fs["track_delta_interval_factor"] >>
        //        track_delta_interval_factor;
    }

    init_blade_model_transformer(
        QVector3D(0, 0, 0), QVector3D(-0.875, 4.021, 1.61),
        QVector3D(0.48, 3.314, 0.857), QVector3D(0.875, 4.021, 1.61), 3.31381,
        4.25303, 1.690, 4.25303, QVector3D(1.094, -0.479, -1.023),
        QVector3D(1.094, -0.423, 0.065), 0.486);

    // TODO CHECK INITIAL STATUS
    //    blade_model_transformer->setInitialPose(1.311, 1.328, 1.247, 0,
    //    0.960, 1.079);
    blade_view_matrix.setToIdentity();
    blade_view_matrix.lookAt(QVector3D(0, 8, 4), QVector3D(0, 1, -0.5),
                             QVector3D(0, 0, 1));

    //    front_wheel_center = QVector3D(-0.062, 6.348, 0);       // 5505pro
    //    front_wheel_center = QVector3D(-0.062, 5.369, 0);       // 3505
    front_wheel_center = config.front_wheel_center;

    left_max_blade_point =
        front_wheel_center + QVector3D(-5.66217, -2.60018, 0);
    right_max_blade_point =
        front_wheel_center + QVector3D(5.66217, -2.60018, 0);

    update();
}

DriveAssistant::~DriveAssistant() {
    delete blade_model_transformer;
}

void DriveAssistant::adjustLeftWheelDeltaTheta(float delta_theta) {
    float theta_bound = 3.141592653 / 4;
    left_wheel_theta += delta_theta;

    if (left_wheel_theta < -theta_bound) {
        left_wheel_theta = -theta_bound;
    } else if (left_wheel_theta > theta_bound) {
        left_wheel_theta = theta_bound;
    }
    update();
}

/**
 * 调整左轮的转向角度
 * 
 * 此函数用于调整驱动装置的左轮转向角度。它将输入的角度值限制在一个预定义的最大偏转角度范围内，
 * 确保轮子不会过度转向，从而保持设备的稳定性。
 * 
 * @param theta 欲调整的左轮转向角度（弧度制）
 */
void DriveAssistant::adjustLeftWheelTheta(float theta) {
    // 定义左轮转向角度的最大偏转值（π/4弧度）
    float theta_bound = 3.141592653 / 4;
    left_wheel_theta  = theta;  // 直接设置左轮转向角度为输入值

    // 检查并调整左轮转向角度，确保其在可接受的范围内
    if (left_wheel_theta < -theta_bound) {
        left_wheel_theta = -theta_bound; // 如果角度小于最小值，则设置为最小值
    } else if (left_wheel_theta > theta_bound) {
        left_wheel_theta = theta_bound; // 如果角度大于最大值，则设置为最大值
    }
    update();  // 调用更新函数，应用新的转向角度
}

/**
 * 更新驱动辅助对象的状态。
 * 此函数计算并更新与车辆转向相关的所有轨迹和半径信息，
 * 包括前轮中心、左右轮的转向角度、左右轮的半径、左右铲刀轨迹、安全轨迹等。
 * 该更新基于当前前轮中心位置、左右轮的转向角度以及车辆配置信息。
 */
void DriveAssistant::update() {
    // 设置原点到前轮中心的垂直距离
    circle_origin.setY(front_wheel_center.y() - wheel_h_dis);
    float eps = 1e-4;   // 定义一个极小值，用于处理角度接近0的情况

    // 确保左轮转向角度不为0，以避免除以0的情况
    if (abs(left_wheel_theta) < eps) {
        left_wheel_theta = eps;
        // 在左轮转向角度为0时，计算并设置原点的X坐标
        circle_origin.setX(front_wheel_center.x() - wheel_w_dis / 2 -
                           left_radius);
    }

    // 根据左轮转向角度的正负，更新转向方向和左右轮的半径信息
    if (left_wheel_theta > 0) {
        turning_direction = 1;
        left_radius       = abs(wheel_h_dis / sinf(left_wheel_theta));
        // 计算并设置左轮的圆心位置
        circle_origin.setX(front_wheel_center.x() +
                           left_radius * cos(left_wheel_theta) -
                           wheel_w_dis / 2);
        // 计算右轮的转向角度和半径
        float tp          = left_radius * cos(left_wheel_theta) - wheel_w_dis;
        right_wheel_theta = atan(wheel_h_dis / tp);
        right_radius      = abs(wheel_h_dis / sin(right_wheel_theta));
    } else {
        turning_direction = -1;
        left_radius       = abs(wheel_h_dis / sinf(left_wheel_theta));
        // 计算并设置左轮的圆心位置
        circle_origin.setX(front_wheel_center.x() - wheel_w_dis / 2 -
                           left_radius * cos(left_wheel_theta));
        // 计算右轮的转向角度和半径
        float tp          = left_radius * cos(left_wheel_theta) + wheel_w_dis;
        right_wheel_theta = -atan(wheel_h_dis / tp);
        right_radius      = abs(wheel_h_dis / sin(right_wheel_theta));
    }

    // 计算中轮半径，用于左右轨迹的设置
    float mid_radius = (left_radius + right_radius) / 2;
    // 将圆心位置转换为Eigen向量，用于后续轨迹的设置
    Eigen::Vector3f eigen_circle_origin(circle_origin.x() + track_delta_x,
                                        circle_origin.y() + track_delta_y, 0);
    // 设置左右轨迹、左右铲刀轨迹和安全轨迹的信息
    left_track.set(left_radius, mid_radius, turning_direction,
                   eigen_circle_origin, 0, config.wheel_track_width);
    left_track.setCross(0.00001, abs(left_wheel_theta), this->cross_interval, 1,
                        track_delta_interval_factor);
    back_left_track.set(left_radius, mid_radius, turning_direction,
                        eigen_circle_origin, 0, config.wheel_track_width);
    back_left_track.setCross(0.5, abs(left_wheel_theta), this->cross_interval,
                             1, track_delta_interval_factor);
    // 计算并设置左右铲刀相对于圆心的半径
    Eigen::Vector3f chandao_left_ground, chandao_right_ground;
    chandao_left_ground =
        getBladeAxisToGroundAxis(blade_model_transformer->cur_chandao_left);
    chandao_right_ground =
        getBladeAxisToGroundAxis(blade_model_transformer->cur_chandao_right);
    float chandao_left_radius =
        (chandao_left_ground - eigen_circle_origin).norm();
    float chandao_right_radius =
        (chandao_right_ground - eigen_circle_origin).norm();
    left_blade_track.set(chandao_left_radius, mid_radius, turning_direction,
                         eigen_circle_origin, 0, config.blade_track_width);
    right_blade_track.set(chandao_right_radius, mid_radius, turning_direction,
                          eigen_circle_origin, 0, config.blade_track_width);

    right_track.set(right_radius, mid_radius, turning_direction,
                    eigen_circle_origin, 0, config.wheel_track_width);
    right_track.setCross(0.00001, abs(right_wheel_theta), this->cross_interval,
                         1, track_delta_interval_factor);
    back_right_track.set(right_radius, mid_radius, turning_direction,
                         eigen_circle_origin, 0, config.wheel_track_width);
    back_right_track.setCross(0.5, abs(right_wheel_theta), this->cross_interval,
                              1, track_delta_interval_factor);
    // 计算左右安全半径，用于安全区域的设置
    QVector3D left_safe_radius =
        left_max_blade_point -
        QVector3D(circle_origin.x(), circle_origin.y(), 0);
    QVector3D right_safe_radius =
        right_max_blade_point -
        QVector3D(circle_origin.x(), circle_origin.y(), 0);

    right_safe_track.set(right_safe_radius.length(), mid_radius,
                         turning_direction, eigen_circle_origin, 0,
                         config.blade_track_width);
    left_safe_track.set(left_safe_radius.length(), mid_radius,
                        turning_direction, eigen_circle_origin, 0,
                        config.blade_track_width);
}

/**
 * @brief 设置驾驶辅助系统的参数
 *
 * 本函数用于初始化和更新驾驶辅助系统的参数，这些参数包括：
 * - 轮子的宽度
 * - 轮子的高度
 * - 车轮间的横向间隔
 * - 轨迹的长度
 * - 轨迹的宽度
 * 更新参数后，会调用update()方法进行内部状态的更新。
 *
 * @param wheel_w_dis_ 轮子的宽度
 * @param wheel_h_dis_ 轮子的高度
 * @param cross_interval_ 车轮间的横向间隔
 * @param track_length_ 轨迹的长度
 * @param track_width_ 轨迹的宽度
 */
void DriveAssistant::setDriveAssistant(float wheel_w_dis_, float wheel_h_dis_,
                                       float cross_interval_,
                                       float track_length_,
                                       float track_width_) {
    this->wheel_w_dis    = wheel_w_dis_;
    this->wheel_h_dis    = wheel_h_dis_;
    this->cross_interval = cross_interval_;
    this->track_length   = track_length_;
    this->track_width    = track_width_;

    // 更新内部状态
    this->update();
}

void DriveAssistant::setTrackCross(float width, float loc, float interval,
                                   float len) {
    left_track.setCross(width, loc, interval, len, track_delta_interval_factor);
    right_track.setCross(width, loc, interval, len,
                         track_delta_interval_factor);
}

void DriveAssistant::adjustTrackDelta(float delta_x, float delta_y,
                                      float delta_factor) {
    this->track_delta_x += delta_x;
    this->track_delta_y += delta_y;
    this->track_delta_interval_factor += delta_factor;
    setTrackCross(0.1, 0, this->cross_interval, 1);
    this->update();
}

void DriveAssistant::saveTrackDelta() {
    cv::FileStorage fs;
    if (!fs.open(track_delta_yaml_path, cv::FileStorage::WRITE)) {
        printf("cannot open %s to write\n", track_delta_yaml_path.c_str());
    }
    fs.write("track_delta_x", track_delta_x);
    fs.write("track_delta_y", track_delta_y);
    fs.write("track_delta_interval_factor", track_delta_interval_factor);
}

void DriveAssistant::init_blade_model_transformer(
    const QVector3D& _p_A, const QVector3D& _p_D, const QVector3D& _p_F,
    const QVector3D& _p_G, float _l_AE, float _l_AB, float _l_BC, float _l_AC,
    const QVector3D& _p_2_nC, const QVector3D& _p_2_nA, float _l_nBnC) {
    delete blade_model_transformer;
    blade_model_transformer = new BladeModelTransformer();
}
