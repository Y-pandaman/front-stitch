#include "track.h"
#include <utility>

Track::Track(int _loc, float _radius, float _mid_radius, float _width,
             float _length, Eigen::Vector3f _center, int _segment_num)
    : track_radius(_radius), track_width(_width), track_length(_length),
      mid_radius(_mid_radius), center(std::move(_center)), track_loc(_loc),
      segment_num(_segment_num),
      point_len(TRACK_POS_COORD_LEN + TRACK_TEXTURE_COORD_LEN),
      point_num(_segment_num * 2 * 3) {
    point_list = nullptr;
}

Track::~Track() {
    delete[] point_list;
}

void Track::generatePointList() {
    if (up_to_date)
        return;
    if (point_num_changed) {
        delete point_list;
        point_list = new float[point_num * TRACK_POINT_LEN];
    } else if (point_list == nullptr) {
        point_list = new float[point_num * TRACK_POINT_LEN];
    }
    float len_per_seg   = track_length / (float)segment_num;
    float angle_per_seg = len_per_seg / track_radius;
    float delta_width   = track_width / 2;
    float max_angle     = begin_angle + 1.2 * M_PI;
    for (int i = 0; i < segment_num; i++) {
        for (int j = 0; j < 6; j++) {
            int offset      = (i * 6 + j) * TRACK_POINT_LEN;
            float cur_angle = i * angle_per_seg + begin_angle;
            if (cur_angle > max_angle)
                cur_angle = max_angle;
            if (j == 2 || j == 3 || j == 4)
                cur_angle += angle_per_seg;
            float cur_radius;
            if (j == 0 || j == 5 || j == 4) {
                cur_radius = track_radius + delta_width * orientation;
                point_list[offset + 3] = 1.;
            } else if (j == 1 || j == 2 || j == 3) {
                cur_radius = track_radius - delta_width * orientation;
                point_list[offset + 3] = 0.;
            }
            point_list[offset] =
                cur_radius * cosf(cur_angle) * -orientation + center.x();
            point_list[offset + 1] = cur_radius * sinf(cur_angle) + center.y();
            point_list[offset + 2] = 0;
            point_list[offset + 4] = (cur_angle - cross_loc) * track_radius;
        }
    }
    if (have_cross) {
        int label_id = 0;
        if (have_number_label)
            label_point_list.clear();
        for (int i = 0; i < cross_num; i++) {
            float cur_loc_angle = cross_loc + (float)i * cross_interval *
                                                  interval_factor /
                                                  track_radius;
            for (int j = 0; j < 6; j++) {
                int offset = (segment_num * 6 + i * 6 + j) * TRACK_POINT_LEN;

                float cur_angle;
                if (j == 1 || j == 2 || j == 3)
                    cur_angle = cur_loc_angle;
                else
                    cur_angle = cur_loc_angle + (cross_width / track_radius);

                float cur_radius;
                if (j == 0 || j == 1 || j == 5)
                    cur_radius = track_radius;
                else
                    cur_radius = track_radius +
                                 cross_len * (float)(track_loc * orientation);

                point_list[offset] =
                    cur_radius * cosf(cur_angle) * (float)(-orientation) +
                    center.x();
                point_list[offset + 1] =
                    cur_radius * sinf(cur_angle) + center.y();
                point_list[offset + 2] = 0;
                point_list[offset + 3] = 1;
                point_list[offset + 4] = (cur_angle - cross_loc) * track_radius;
            }
            if (have_number_label && track_loc == -1 && cur_loc_angle > -1e-4) {
                if (label_id == 0) {
                    label_id++;
                    continue;
                }
                Eigen::Vector3f cur_pos, text_dire, text_up;
                float x =
                    track_radius * cosf(cur_loc_angle) * (float)(-orientation) +
                    center.x();
                float y   = track_radius * sinf(cur_loc_angle) + center.y();
                cur_pos   = Eigen::Vector3f(x, y, 0);
                text_dire = track_loc * orientation * (cur_pos - center);
                text_dire.normalize();
                text_up = Eigen::Vector3f(0, 1, 0);
                text_up.normalize();
                // change label direction
                text_up += Eigen::Vector3f(0, 0, 0.8);
                text_up.normalize();
                std::string label_str =
                    std::to_string(label_id * (int)cross_interval);
                int texture_height = 72, texture_width = 157;
                float factor =
                    1.2f / 157.0f * (0.25f * (float)label_id + 0.75f);
                Eigen::Vector3f exact_pos =
                    cur_pos + texture_width * factor * (-text_dire);

                for (int j = 0; j < 6; j++) {
                    Eigen::Vector3f point = exact_pos;
                    float u = 0.0, v = 0.2 * (label_id);
                    if (j == 1 || j == 2 || j == 3) {
                        point += texture_width * factor * text_dire;
                        u = 1.0;
                    }
                    if (j == 2 || j == 3 || j == 4) {
                        point += texture_height * factor * text_up;
                        v -= 0.2;
                    }
                    label_point_list.push_back(point.x());
                    label_point_list.push_back(point.y());
                    label_point_list.push_back(point.z());
                    label_point_list.push_back(u);
                    label_point_list.push_back(v);
                }
                label_id++;
            }
        }
    }
    up_to_date = true;
}

/**
 * @brief 设置交叉线的参数
 *
 * 该函数用于为轨道对象设置交叉线的相关参数，并根据这些参数计算出交叉线的数量和总点数。
 * 该函数还会更新与交叉线相关的内部状态标志。
 *
 * @param width 交叉线的宽度
 * @param loc 交叉线的位置
 * @param interval 交叉线之间的间隔
 * @param len 交叉线的长度
 * @param _interval_factor 交叉线间隔的调整因子
 */
void Track::setCross(float width, float loc, float interval, float len,
                     float _interval_factor) {
    // 设置交叉线的基本参数
    cross_width    = width;
    cross_loc      = loc;
    cross_len      = len;
    cross_interval = interval;

    // 根据是否已有交叉线来更新点数是否变化的标志
    if (have_cross)
        point_num_changed = false;
    else {
        point_num_changed = true;
        have_cross        = true;
    }

    // 标记参数更新状态
    up_to_date            = false;
    this->interval_factor = _interval_factor;

    // 计算交叉线数量，并据此更新点数
    cross_num =
        (int)((track_length - (cross_loc - begin_angle) * track_radius) /
              cross_interval);
    // 计算总点数，包括段点和交叉点
    point_num = segment_num * 6 + cross_num * 6;
}

/**
 * 设置轨迹的属性
 *
 * 该函数用于初始化或更新一个轨迹对象的属性，包括半径、中间半径、原始方向，
 * 中心位置、起始角度和轨迹宽度。
 *
 * @param _radius         轨迹的外半径。
 * @param _mid_radius     轨迹的中间半径。
 * @param _ori            轨迹的原始方向，通常是一个整数标识。
 * @param _center         轨迹的中心位置，使用Eigen::Vector3f表示。
 * @param _begin_angle    轨迹的起始角度，通常以弧度为单位。
 * @param _width          轨迹的宽度。
 */
void Track::set(float _radius, float _mid_radius, int _ori,
                const Eigen::Vector3f& _center, float _begin_angle,
                float _width) {
    this->mid_radius   = _mid_radius;    // 设置中间半径
    this->begin_angle  = _begin_angle;   // 设置起始角度
    this->track_radius = _radius;        // 设置外半径
    this->center       = _center;        // 设置中心位置
    this->orientation  = _ori;           // 设置原始方向
    this->up_to_date   = false;          // 标记为未更新状态
    this->track_width  = _width;         // 设置轨迹宽度
}

float* Track::getPointList() {
    if (!this->up_to_date) {
        generatePointList();
    } else {
        printf("up to date!!!!!!!\n");
    }
    return this->point_list;
}

uint Track::getBufferSize() const {
    return point_num * TRACK_POINT_LEN * sizeof(float);
}

void Track::print_point_list(std::string filename) {
    FILE* fp = fopen(filename.c_str(), "w");
    fprintf(fp, "[");
    for (int i = 0; i < point_num; i++) {
        fprintf(fp, "[");
        int p = i * TRACK_POINT_LEN;
        fprintf(fp, "%f", point_list[p]);
        for (int j = 1; j < TRACK_POINT_LEN; j++) {
            fprintf(fp, ", %f", point_list[p + j]);
        }
        fprintf(fp, "],\n");
    }
    fprintf(fp, "]");
    fclose(fp);
}

void Track::setTrack(int _loc, float _radius, float _mid_radius, float _width,
                     float _length, const Eigen::Vector3f& _center,
                     int _segment_num) {
    this->track_radius = _radius;
    this->track_width  = _width;
    this->track_length = _length;
    this->mid_radius   = _mid_radius;
    this->center       = _center;
    this->track_loc    = _loc;
    this->segment_num  = _segment_num;
    this->point_len    = TRACK_POINT_LEN;
    this->point_num    = _segment_num * 2 * 3;

    this->up_to_date        = false;
    this->have_cross        = false;
    this->have_number_label = true;
    this->point_num_changed = true;
}
