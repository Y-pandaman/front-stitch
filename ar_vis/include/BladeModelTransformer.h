#ifndef CYLINDER_STITCHER_BLADEMODELTRANSFORMER_H
#define CYLINDER_STITCHER_BLADEMODELTRANSFORMER_H

#include "Config.h"
#include "Model.h"
#include <Eigen/Eigen>
#include <QMatrix4x4>
#include <QVector2D>
#include <QVector3D>
#include <QVector4D>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>

float getWorldZ(const Eigen::Matrix4f trans, const Vertex& vertex);

class BladeModelTransformer {
public:
    static Eigen::Vector3f transform(Eigen::Matrix4f m, Eigen::Vector3f v) {
        Eigen::Vector4f r = m * Eigen::Vector4f(v.x(), v.y(), v.z(), 1.0);
        return Eigen::Vector3f(r.x() / r.w(), r.y() / r.w(), r.z() / r.w());
    }
    BladeModelTransformer(const QVector3D& _p_A, const QVector3D& _p_D,
                          const QVector3D& _p_F, const QVector3D& _p_G,
                          float _l_AE, float _l_AB, float _l_BC, float _l_AC,
                          const QVector3D& _p_2_nC, const QVector3D& _p_2_nA,
                          float _l_nBnC);
    BladeModelTransformer();
    static std::vector<QVector3D> trilateration(const QVector3D& P1,
                                                const QVector3D& P2,
                                                const QVector3D& P3, float r1,
                                                float r2, float r3);
    static Eigen::Vector3f trilateration(const Eigen::Vector3f& P1,
                                         const Eigen::Vector3f& P2,
                                         const Eigen::Vector3f& P3, float r1,
                                         float r2, float r3);

    int getLowestZ();

    // 平面上两圆相交
    static bool getIntersectionOfTwoCircles(const QVector2D& c0,
                                            const QVector2D& c1, float r0,
                                            float r1,
                                            std::vector<QVector2D>& result);

    static bool intersection_two_circles(const Eigen::Vector2f& c0,
                                         const Eigen::Vector2f& c1, float r0,
                                         float r1,
                                         std::vector<Eigen::Vector2f>& result);

    void setCurrentPoseNew(float l_CG, float l_BD, float l_EF, float rotation,
                           float l_IJ, float l_LM);

    QMatrix4x4 model_matrix_part_0, model_matrix_part_1;

    Model qianyinjia_mesh {}, yuanpan_mesh {}, chandaozhijia_mesh {},
        chandao_mesh {};
    Model yougang1_down_mesh {}, yougang1_up_mesh {};
    Model yougang2_down_mesh {}, yougang2_up_mesh {};
    Model yougang3_down_mesh {}, yougang3_up_mesh {};
    Model chandaobeimian_mesh {};
    Model static_mesh_1 {}, static_mesh_2 {}, static_mesh_3 {},
        static_mesh_4 {};

    Eigen::Vector3f cur_chandao_left  = Eigen::Vector3f(1348., -3267., 21.);
    Eigen::Vector3f cur_chandao_right = Eigen::Vector3f(1156., 4003., -1186.);

    float chandao_left_z, chandao_right_z;
    float chandao_set_left_z = -1e5, chandao_set_right_z = -1e5;

    /**
     * 将当前姿态的左右高度设置为零高度
     * 该函数不接受参数，也不返回任何值。
     * 主要作用是将机器人左右两侧的高度基准点重置为初始状态。
     */
    void setCurrentPoseZeroHeight() {
        // 将当前左右两侧的高度值分别赋值给对应的设置值变量
        chandao_set_left_z  = chandao_left_z;
        chandao_set_right_z = chandao_right_z;
    }

    /**
     * 获取左右两侧相对于设定高度的相对高度
     *
     * @param left_height_ 引用，用于存储左侧高度相对于设定高度的差值
     * @param right_height_ 引用，用于存储右侧高度相对于设定高度的差值
     */
    void getRelativeHeight(float& left_height_, float& right_height_) {
        // 计算左侧高度相对于设定高度的差值
        left_height_ = chandao_left_z - chandao_set_left_z;
        // 计算右侧高度相对于设定高度的差值
        right_height_ = chandao_right_z - chandao_set_right_z;
    }

private:
    // 世界坐标系下
    const QVector3D p_A, p_D, p_F, p_G;

    const float l_AE {}, l_AB {}, l_BC {}, l_AC {};

    // 铲刀坐标系下
    const QVector3D p_2_nC, p_2_nA;
    const float l_nBnC {};

    Eigen::Vector3f A =
        Eigen::Vector3f(-2082.45 / 1000.0, 0.25 / 1000.0, 469.8826 / 1000.0);
    Eigen::Vector3f D =
        Eigen::Vector3f(568.9 / 1000.0, 898.765 / 1000.0, 1599.34 / 1000.0);
    Eigen::Vector3f G = Eigen::Vector3f(570.6078 / 1000.0, -889.053 / 1000.0,
                                        1592.245 / 1000.0);
    Eigen::Vector3f F = Eigen::Vector3f(929.3075 / 1000.0, -181.493 / 1000.0,
                                        977.7199 / 1000.0);
    Eigen::Vector3f init_E =
        Eigen::Vector3f(940.0547 / 1000.0, 825.25 / 1000.0, 542.8826 / 1000.0);
    Eigen::Vector3f init_B =
        Eigen::Vector3f(639.5486 / 1000.0, 825.25 / 1000.0, 542.826 / 1000.0);
    Eigen::Vector3f init_C =
        Eigen::Vector3f(639.5486 / 1000.0, -824.75 / 1000.0, 542.8826 / 1000.0);
    Eigen::Vector3f H =
        Eigen::Vector3f(481.1545 / 1000.0, 0 / 1000.0, 0 / 1000.0);
    Eigen::Vector3f I =
        Eigen::Vector3f(1738.549 / 1000.0, -5.0582 / 1000.0, 47.8826 / 1000.0);
    Eigen::Vector3f init_J =
        Eigen::Vector3f(1246.574 / 1000.0, -5.0582 / 1000.0, 20.3631 / 1000.0);
    Eigen::Vector3f K =
        Eigen::Vector3f(1313.574 / 1000.0, -5.0582 / 1000.0, -437.801 / 1000.0);
    Eigen::Vector3f L =
        Eigen::Vector3f(1219.351 / 1000.0, -927.75 / 1000.0, -227.983 / 1000.0);
    Eigen::Vector3f init_M = Eigen::Vector3f(
        1219.351 / 1000.0, 1846.618 / 1000.0, -227.983 / 1000.0);

    Eigen::Matrix4f qianyinjia_local    = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f init_yuanpan_local  = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f chandaozhijia_local = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f chandao_local       = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f qianyinjia_local_estimated = Eigen::Matrix4f::Identity();

    Eigen::Vector3f yougang1_y_axis = Eigen::Vector3f::Zero();
    Eigen::Vector3f yougang2_y_axis = Eigen::Vector3f::Zero();
    Eigen::Vector3f yougang3_y_axis = Eigen::Vector3f::Zero();

    Eigen::Matrix4f qianyinjia_world    = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f yuanpan_world       = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f chandaozhijia_world = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f chandao_world       = Eigen::Matrix4f::Identity();

    Eigen::Vector3f chandao_left  = Eigen::Vector3f(1348., -3267., 21.);
    Eigen::Vector3f chandao_right = Eigen::Vector3f(1156., 4003., -1186.);
};

#endif   // CYLINDER_STITCHER_BLADEMODELTRANSFORMER_H
