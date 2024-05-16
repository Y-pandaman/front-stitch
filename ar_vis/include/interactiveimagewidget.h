#ifndef INTERACTIVEIMAGEWIDGET_H
#define INTERACTIVEIMAGEWIDGET_H

#include "driveassistant.h"
#include "innoreal_timer.hpp"
#include "lane_line.pb.h"
#include "mypainter.h"
#include "shaders.h"
#include <Eigen/Eigen>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QObject>
#include <QOffscreenSurface>
#include <QVector3D>
#include <atomic>
#include <ecal/ecal.h>
#include <ecal/msg/protobuf/subscriber.h>
#include <opencv2/opencv.hpp>

struct AssistantTransformInfo {
    QMatrix4x4 view, projection, model;
    AssistantTransformInfo() {
        view.setToIdentity();
        projection.setToIdentity();
        model.setToIdentity();
    }
};

class InteractiveImageWidget : public QOpenGLWidget,
                               protected QOpenGLExtraFunctions {
    Q_OBJECT
public:
    InteractiveImageWidget(int width, int height, float fx, float fy, float cx,
                           float cy, Eigen::Matrix4f camera_pose,
                           Eigen::Matrix4f plane_pose,
                           QWidget* parent = nullptr, int blade_width = 0,
                           int blade_height = 0);
    virtual ~InteractiveImageWidget() { }
    void initializeGL();

    virtual void paintGL() Q_DECL_OVERRIDE;

    enum GetImageFrom { FB_TRACK, FB_BLADE, FB_BACK_TRACK };

    bool updateTrackInCudaMem();
    float* getTrackFBCudaPtr();

    // 3channels
    void getImageCPU(cv::Mat& image, GetImageFrom from);
    // 4channels
    void getImageCPU4Channels(cv::Mat& image, GetImageFrom from);

    void setWheelTheta(float theta);

    void setDriveAssistant(float wheel_w_dis_, float wheel_h_dis_,
                           float cross_interval_, float track_length_,
                           float track_width_);

    void setCameraGround(const cv::Mat& mat_cam3_to_tag_,
                         const cv::Mat& mat_cam1_to_cam3_);

    void setCameraGround(const cv::Mat& mat_cam1_to_tag_);

    void setTrackCross(float width, float loc, float interval, float len);

    void adjustTrackDelta(float delta_x, float delta_y, float delta_factor);

        /**
     * @brief 设置键盘输入状态
     * @param flag true 表示启用键盘输入，false 表示禁用键盘输入
     */
    void setKeyBoardInput(bool flag) {
        key_board_input = flag;
    }

    /**
     * @brief 设置前轮驱动线状态
     * @param flag true 表示启用前轮驱动线，false 表示禁用前轮驱动线
     */
    void setFrontWheelDrivingLine(bool flag) {
        front_wheel_driving_line = flag;
    }

    /**
     * @brief 设置刀片操作线状态
     * @param flag true 表示启用刀片操作线，false 表示禁用刀片操作线
     */
    void setBladeOperationLine(bool flag) {
        blade_operation_line = flag;
    }

    /**
     * @brief 设置刀片最大操作线状态
     * @param flag true 表示启用刀片最大操作线，false 表示禁用刀片最大操作线
     */
    void setBladeMaxOperationLine(bool flag) {
        blade_max_operation_line = flag;
    }

    /**
     * @brief 设置绘制刀片模型状态
     * @param flag true 表示启用绘制刀片模型，false 表示禁用绘制刀片模型
     */
    void setDrawBladeModel(bool flag) {
        flag_draw_blade_model = flag;
    }

    void setBladeCurrentPose(float l_CG, float l_BD, float l_EF, float theta,
                             float l_IJ, float l_LM);

    int getBladeLowestZ();

    DriveAssistant* drive_assistant;

protected:
    void mouseMoveEvent(QMouseEvent* event) Q_DECL_OVERRIDE;

    void wheelEvent(QWheelEvent* event) Q_DECL_OVERRIDE;

    void mouseDoubleClickEvent(QMouseEvent* event) Q_DECL_OVERRIDE;

    void keyPressEvent(QKeyEvent* event) Q_DECL_OVERRIDE;

    void adjustTranslation(const QPointF& delta);

    void adjustScale(float delta);

    void updateImageTranslation();

    QPointF mouse_prev_pos;
    QPointF image_translation;
    float image_scale = 1;

private:
    AssistantTransformInfo assistant_transform_info;
    FrameBuffer *fb_track_ = NULL, *fb_blade_ = nullptr,
                *fb_back_track_ = nullptr;
    MyPainter* my_painter_      = NULL;
    QMatrix4x4 transform_matrix;

    bool show_drive_assistant = true;

    int width_, height_;
    int blade_width_, blade_height_;
    float fx_, fy_, cx_, cy_;
    Eigen::Matrix4f camera_pose_, plane_pose_;
    enum UpdateAssistantTransformInfoBy {
        BY_CAM1_CAM3_TAG,
        BY_CAM1_TAG
    } update_assistant_transform_info_by;
    void updateAssistantTransformInfo();

    QMatrix4x4 cam3_to_tag, cam1_to_cam3, cam1_to_tag;
    bool assistant_inited = false;

    std::atomic_bool front_wheel_driving_line  = true,
                     blade_operation_line      = true,
                     blade_max_operation_line  = true,
                     flag_draw_back_track_line = false;
    std::atomic_bool flag_draw_blade_model     = true;

    std::atomic_bool blade_input_set_zero_height = false;

    std::shared_ptr<eCAL::protobuf::CSubscriber<xcmg_proto::lane_line>>
        m_sub_lane_line;

    void ecalLaneLineGetCallBack(const char* _topic_name,
                                 const xcmg_proto::lane_line& msg,
                                 long long _time, long long _clock,
                                 long long _id);

    bool key_board_input = false;

    void saveDriveAssistantTrackDelta();
};

#endif   // INTERACTIVEIMAGEWIDGET_H
