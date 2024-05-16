#include "interactiveimagewidget.h"

/**
 * 构造函数：InteractiveImageWidget
 * 用于创建一个交互式图像小部件，支持OpenGL渲染。
 *
 * @param width 图像宽度。
 * @param height 图像高度。
 * @param fx 焦距x。
 * @param fy 焦距y。
 * @param cx 像素中心x。
 * @param cy 像素中心y。
 * @param camera_pose 相机在世界坐标系中的姿态矩阵。
 * @param plane_pose 平面在世界坐标系中的姿态矩阵。
 * @param parent 父QWidget。
 * @param blade_width 铲刀宽度，如果为0，则与图像宽度相同。
 * @param blade_height 铲刀高度，如果为0，则与图像高度相同。
 */
InteractiveImageWidget::InteractiveImageWidget(int width, int height, float fx,
                                               float fy, float cx, float cy,
                                               Eigen::Matrix4f camera_pose,
                                               Eigen::Matrix4f plane_pose,
                                               QWidget* parent, int blade_width,
                                               int blade_height)
    : QOpenGLWidget(parent) {
    // 翻转相机的y轴和z轴，以适应OpenGL的坐标系。
    camera_pose.col(1) *= -1;
    camera_pose.col(2) *= -1;

    // 初始化宽度和高度。
    width_  = width;
    height_ = height;
    // 如果铲刀尺寸未指定，则默认与图像尺寸相同。
    if (blade_width == 0 || blade_height == 0) {
        blade_width_  = width;
        blade_height_ = height;
    } else {
        blade_width_  = blade_width;
        blade_height_ = blade_height;
    }
    // 初始化相机参数。
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    // 保存传入的相机和平面姿态矩阵。
    camera_pose_ = camera_pose;
    plane_pose_  = plane_pose;
    // 如果像素中心未指定，则默认为图像中心。
    if (cx < 0 || cy < 0) {
        cx = (width - 1) * 0.5;
        cy = (height - 1) * 0.5;
    }

    // 启用鼠标追踪和键盘捕获。
    setMouseTracking(true);
    grabKeyboard();

    // 初始化相机间和相机到标签的转换矩阵。
    cam1_to_cam3.setToIdentity();
    cam3_to_tag.setToIdentity();

    // 如果eCAL未初始化，则进行初始化。
    if (!eCAL::IsInitialized())
        eCAL::Initialize();
    // 订阅Lane_Line_msg主题。
    m_sub_lane_line =
        std::make_shared<eCAL::protobuf::CSubscriber<xcmg_proto::lane_line>>(
            "Lane_Line_msg");

    // 绑定接收回调函数。
    auto cb = std::bind(&InteractiveImageWidget::ecalLaneLineGetCallBack, this,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        std::placeholders::_5);
    m_sub_lane_line->AddReceiveCallback(cb);
}

/**
 * 设置互动图像小部件的轮子转角
 *
 * 该函数用于调整互动图像小部件中左轮的转动角度。通过驱动助手来实现具体的调整操作。
 *
 * @param theta 指定左轮的新转角，单位为弧度。
 */
void InteractiveImageWidget::setWheelTheta(float theta) {
    // 通过驱动助手调整左轮的转角
    drive_assistant->adjustLeftWheelTheta(theta);
}

void InteractiveImageWidget::initializeGL() {
    makeCurrent();
    initializeOpenGLFunctions();

    if (fb_track_ == NULL || fb_blade_ == nullptr ||
        fb_back_track_ == nullptr) {
        if (fb_track_ == NULL)
            fb_track_ = new FrameBuffer;
        if (fb_blade_ == nullptr)
            fb_blade_ = new FrameBuffer;
        if (fb_back_track_ == nullptr) {
            // printf("new fb_back_track_ = new FrameBuffer\n");
            fb_back_track_ = new FrameBuffer;
        }

        fb_track_->resize(GL_RGBA32F, width_, height_, true, true);
        fb_blade_->resize(GL_RGBA32F, blade_width_, blade_height_, false, true);
        fb_back_track_->resize(GL_RGBA32F, config.back_track_width,
                               config.back_track_height, false, true);

        my_painter_ = new MyPainter(fb_track_, fb_blade_, fb_back_track_, this);
        drive_assistant = my_painter_->getDriveAssistant();
        my_painter_->initializeDrawTrack();
        my_painter_->initializeDrawText();

        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->static_mesh_1);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->static_mesh_2);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->static_mesh_3);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->static_mesh_4);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->qianyinjia_mesh);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->yougang1_up_mesh);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->yougang1_down_mesh);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->yougang2_up_mesh);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->yougang2_down_mesh);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->yougang3_up_mesh);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->yougang3_down_mesh);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->yuanpan_mesh);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->chandaozhijia_mesh);
        my_painter_->initializeDrawModel(
            drive_assistant->blade_model_transformer->chandao_mesh);
    }
    glCheckError();
}

void InteractiveImageWidget::mouseMoveEvent(QMouseEvent* event) {
#if 0
    QPointF delta = event->position() - mouse_prev_pos;

    if (event->buttons() & Qt::LeftButton) {
      adjustTranslation(delta / image_scale);
      updateImageTranslation();
      update();
    }
    mouse_prev_pos = event->pos();
#endif
}

void InteractiveImageWidget::wheelEvent(QWheelEvent* event) {
#if 0
    float delta_scale = event->angleDelta().y() / 12000.0;
    QPointF mouse_pos = event->position();
    QPointF widget_center = QPointF((float)this->width() / 2, (float)this->height() / 2);
    QPointF image_center = widget_center + image_translation;
    QPointF delta_trans = mouse_pos - image_center;

    adjustScale(delta_scale);
    adjustTranslation(-delta_trans * delta_scale);
    updateImageTranslation();

    update();
#endif
}

void InteractiveImageWidget::mouseDoubleClickEvent(QMouseEvent* event) {
    image_translation = QPointF(0, 0);
    image_scale       = 1;

    updateImageTranslation();

    update();
}

void InteractiveImageWidget::keyPressEvent(QKeyEvent* event) {
    glCheckError();
    if (!key_board_input)
        return;
    if (event->key() == Qt::Key_Left) {
        drive_assistant->adjustLeftWheelDeltaTheta(-0.01);
    } else if (event->key() == Qt::Key_Right) {
        drive_assistant->adjustLeftWheelDeltaTheta(0.01);
    } else if (event->key() == Qt::Key_V) {
        QOpenGLFunctions* gl_function_ptr =
            QOpenGLContext::currentContext()->functions();

        GLint major, minor;
        gl_function_ptr->glGetIntegerv(GL_MAJOR_VERSION, &major);
        gl_function_ptr->glGetIntegerv(GL_MINOR_VERSION, &minor);

        qDebug() << "OpenGL Version Info: "
                 << (const char*)gl_function_ptr->glGetString(GL_VERSION);
        qDebug() << "OpenGL Version Major: " << major << "Minor :" << minor;
    } else if (event->key() == Qt::Key_A) {
        this->adjustTrackDelta(-0.1, 0, 0);
    } else if (event->key() == Qt::Key_D) {
        this->adjustTrackDelta(0.1, 0, 0);
    } else if (event->key() == Qt::Key_W) {
        this->adjustTrackDelta(0, 0.1, 0);
    } else if (event->key() == Qt::Key_S) {
        this->adjustTrackDelta(0, -0.1, 0);
    } else if (event->key() == Qt::Key_R) {
        this->adjustTrackDelta(0, 0, 0.02);
    } else if (event->key() == Qt::Key_F) {
        this->adjustTrackDelta(0, 0, -0.02);
    } else if (event->key() == Qt::Key_O) {
        this->drive_assistant->saveTrackDelta();
    }
    update();
}

void InteractiveImageWidget::adjustTranslation(const QPointF& delta) {
    image_translation += delta;
}

void InteractiveImageWidget::adjustScale(float delta) {
    image_scale *= 1 + delta;
}

void InteractiveImageWidget::updateImageTranslation() {
    transform_matrix.setToIdentity();
    // widgt size -> [-1, 1]
    transform_matrix.translate(image_translation.x() * 2.0 / this->width(),
                               -image_translation.y() * 2.0 / this->height());

    transform_matrix.scale(image_scale, image_scale);
}
void InteractiveImageWidget::paintGL() {
    static int paint_count = 0;

    if (!assistant_inited) {
        this->updateAssistantTransformInfo();
        assistant_inited = true;
        glCheckError();
    }

    fb_track_->bind();
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    fb_track_->unbind();

    fb_blade_->bind();
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    fb_blade_->unbind();

    fb_back_track_->bind();
    glClearColor(0., 0., 0., 0.);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    fb_back_track_->unbind();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    if (blade_input_set_zero_height) {
        blade_input_set_zero_height = false;
        drive_assistant->blade_model_transformer->setCurrentPoseZeroHeight();
    }

    if (show_drive_assistant) {
        if (blade_max_operation_line) {
            QVector3D color(0x24, 0xff, 0x00);
            color /= 255.0f;
            my_painter_->drawTrack(drive_assistant->left_safe_track,
                                   assistant_transform_info.projection,
                                   assistant_transform_info.view,
                                   assistant_transform_info.model, color, 1.1,
                                   false, 0);
            my_painter_->drawTrack(drive_assistant->right_safe_track,
                                   assistant_transform_info.projection,
                                   assistant_transform_info.view,
                                   assistant_transform_info.model, color, 1.1,
                                   false, 0);
        }
        if (blade_operation_line) {
            my_painter_->drawTrack(
                drive_assistant->left_blade_track,
                assistant_transform_info.projection,
                assistant_transform_info.view, assistant_transform_info.model,
                QVector3D(0.5686, 0.9333, 0.9019), 1, false, 0);
            my_painter_->drawTrack(
                drive_assistant->right_blade_track,
                assistant_transform_info.projection,
                assistant_transform_info.view, assistant_transform_info.model,
                QVector3D(0.5686, 0.9333, 0.9019), 1, false, 0);
        }
        if (front_wheel_driving_line) {
            QVector3D lane_line_color(0, 255, 255);
            lane_line_color /= 255.0;

            float min_show_v = config.track_min_show_v;
            my_painter_->drawTrack(
                drive_assistant->left_track,
                assistant_transform_info.projection,
                assistant_transform_info.view, assistant_transform_info.model,
                lane_line_color, min_show_v, true,
                (float)paint_count / 100.0 * config.wheel_track_speed);
            my_painter_->drawTrack(
                drive_assistant->right_track,
                assistant_transform_info.projection,
                assistant_transform_info.view, assistant_transform_info.model,
                lane_line_color, min_show_v, true,
                (float)paint_count / 100.0 * config.wheel_track_speed);
            my_painter_->drawText(
                drive_assistant->left_track.getLabelPointList(),
                assistant_transform_info.projection,
                assistant_transform_info.view, assistant_transform_info.model,
                lane_line_color);
        }

        if (flag_draw_blade_model) {
            QMatrix4x4 projection;
            projection.setToIdentity();
            projection.perspective(
                60, (float)blade_width_ / (float)blade_height_, 0.1, 1000);

            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->static_mesh_1,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->static_mesh_1
                    .getModelQMatrix(),
                config.blade_color_0);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->static_mesh_2,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->static_mesh_2
                    .getModelQMatrix(),
                config.blade_color_0);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->static_mesh_3,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->static_mesh_3
                    .getModelQMatrix(),
                config.blade_color_0);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->static_mesh_4,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->static_mesh_4
                    .getModelQMatrix(),
                config.blade_color_0);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->qianyinjia_mesh,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->qianyinjia_mesh
                    .getModelQMatrix(),
                config.blade_color_0);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->yougang1_up_mesh,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->yougang1_up_mesh
                    .getModelQMatrix(),
                config.blade_color_0);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->yougang1_down_mesh,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->yougang1_down_mesh
                    .getModelQMatrix(),
                config.blade_color_1);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->yougang2_up_mesh,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->yougang2_up_mesh
                    .getModelQMatrix(),
                config.blade_color_0);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->yougang2_down_mesh,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->yougang2_down_mesh
                    .getModelQMatrix(),
                config.blade_color_1);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->yougang3_up_mesh,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->yougang3_up_mesh
                    .getModelQMatrix(),
                config.blade_color_1);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->yougang3_down_mesh,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->yougang3_down_mesh
                    .getModelQMatrix(),
                config.blade_color_0);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->yuanpan_mesh,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->yuanpan_mesh
                    .getModelQMatrix(),
                config.blade_color_0);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->chandaozhijia_mesh,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->chandaozhijia_mesh
                    .getModelQMatrix(),
                config.blade_color_0);
            my_painter_->drawModel(
                drive_assistant->blade_model_transformer->chandao_mesh,
                projection, drive_assistant->blade_view_matrix,
                drive_assistant->blade_model_transformer->chandao_mesh
                    .getModelQMatrix(),
                config.blade_color_0);
        }

        QVector3D lane_line_color(0, 255, 255);
        lane_line_color /= 255.0;
        my_painter_->drawBackTrack(
            drive_assistant->back_left_track,
            assistant_transform_info.projection, assistant_transform_info.view,
            assistant_transform_info.model, lane_line_color, -100);
        my_painter_->drawBackTrack(
            drive_assistant->back_right_track,
            assistant_transform_info.projection, assistant_transform_info.view,
            assistant_transform_info.model, lane_line_color, -100);
        glCheckError();
    } else {
    }

    // painter.endNativePainting();
    QOpenGLExtraFunctions* f =
        QOpenGLContext::currentContext()->extraFunctions();
    fb_track_->bliteFboMSToFbo(f);
    fb_blade_->bliteFboMSToFbo(f);   // TODO CHECK
    fb_back_track_->bliteFboMSToFbo(f);
    glCheckError();

    paint_count++;
}

/**
 * 从指定的帧缓冲区获取图像数据，并将其转换为OpenCV的Mat格式
 * @param image 引用，用于存储从帧缓冲区获取到的图像
 * @param from 指定获取图像的来源，可以是追踪帧缓冲区、铲刀帧缓冲区或后追踪帧缓冲区
 */
void InteractiveImageWidget::getImageCPU(cv::Mat& image, GetImageFrom from) {
    FrameBuffer* fb;
    // 根据指定的来源选择相应的帧缓冲区
    if (from == GetImageFrom::FB_TRACK)
        fb = fb_track_;
    else if (from == GetImageFrom::FB_BLADE)
        fb = fb_blade_;
    else if (from == GetImageFrom::FB_BACK_TRACK) {
        fb = fb_back_track_;
    } else {
        // 如果指定的来源无效，则打印错误信息并退出程序
        printf("getImageCPU error!\n");
        exit(0);
    }

    // 初始化图像矩阵，准备接收帧缓冲区的数据
    image = cv::Mat::zeros(fb->height_, fb->width_, CV_32FC4);
    // 从帧缓冲区下载纹理数据到image矩阵
    fb->downloadTexture(image.data, GL_RGBA, GL_FLOAT);

    // 翻转图像，使其上下颠倒
    cv::flip(image, image, 0);
    // 将图像颜色空间从BGR转换为RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    // 将图像数据类型从32位浮点数转换为8位无符号整数，并调整其尺度
    image.convertTo(image, CV_8UC3, 255.0);
}

/**
 * 从指定的帧缓冲区获取4通道图像数据，并将其保存到传入的OpenCV Mat对象中。
 *
 * @param image 引用，用于存储从帧缓冲区获取的图像数据的OpenCV Mat对象。
 * @param from 一个枚举值，指定要从哪个帧缓冲区获取图像数据。
 */
void InteractiveImageWidget::getImageCPU4Channels(cv::Mat& image,
                                                  GetImageFrom from) {
    FrameBuffer* fb;
    // 根据from参数选择对应的帧缓冲区
    if (from == GetImageFrom::FB_TRACK)
        fb = fb_track_;
    else if (from == GetImageFrom::FB_BLADE)
        fb = fb_blade_;
    else if (from == GetImageFrom::FB_BACK_TRACK) {
        fb = fb_back_track_;
    } else {
        // 如果from参数无匹配，打印错误信息并退出程序
        printf("getImageCPU error!\n");
        exit(0);
    }

    // 初始化image为零矩阵，准备存储帧缓冲区下载的图像数据
    image = cv::Mat::zeros(fb->height_, fb->width_, CV_32FC4);
    // 从帧缓冲区下载图像数据到image
    fb->downloadTexture(image.data, GL_RGBA, GL_FLOAT);
    // 翻转图像，使图像上下颠倒
    cv::flip(image, image, 0);
    // 将图像颜色空间从BGRA转换为RGBA
    cv::cvtColor(image, image, cv::COLOR_BGRA2RGBA);
    // 将图像数据类型从浮点型转换为8位无符号整型，并调整其范围至[0, 255]
    image.convertTo(image, CV_8UC4, 255.0);
}

static Eigen::Vector3f
GetIntersectWithLineAndPlane(Eigen::Vector3f point, Eigen::Vector3f direct,
                             Eigen::Vector3f planeNormal,
                             Eigen::Vector3f planePoint) {
    float d = (planePoint - point).dot(planeNormal) / direct.dot(planeNormal);
    return d * direct.normalized() + point;
}

void InteractiveImageWidget::updateAssistantTransformInfo() {
    float z_near = 0.1f, z_far = 1000.0f;

    Eigen::Vector3f plane_pos    = plane_pose_.topRightCorner(3, 1);
    Eigen::Vector3f plane_normal = plane_pose_.topLeftCorner(3, 3).col(2);
    plane_normal.normalize();
    Eigen::Vector3f ray_pos = camera_pose_.topRightCorner(3, 1);
    Eigen::Vector3f ray_dir = plane_normal;

    Eigen::Vector3f intersect_point =
        GetIntersectWithLineAndPlane(ray_pos, ray_dir, plane_normal, plane_pos);
    plane_pose_.topRightCorner(3, 1) = intersect_point;

    camera_pose_                = plane_pose_.inverse() * camera_pose_;
    Eigen::Matrix4f view_matrix = camera_pose_.inverse();

    Eigen::Matrix4f projection_matrix = Eigen::Matrix4f::Zero();
    projection_matrix(0, 0)           = 2.0f * fx_ / fb_track_->width_;
    projection_matrix(1, 1)           = 2.0f * fy_ / fb_track_->height_;
    projection_matrix(0, 2)           = -(2 * cx_ / fb_track_->width_ - 1.0f);
    projection_matrix(1, 2)           = (2 * cy_ / fb_track_->height_ - 1.0f);
    projection_matrix(2, 2)           = -(z_far + z_near) / (z_far - z_near);
    projection_matrix(2, 3) = -(2 * z_far * z_near) / (z_far - z_near);
    projection_matrix(3, 2) = -1;

    memcpy(assistant_transform_info.projection.data(), projection_matrix.data(),
           sizeof(float) * 16);
    memcpy(assistant_transform_info.view.data(), view_matrix.data(),
           sizeof(float) * 16);
    glCheckError();

    if (update_assistant_transform_info_by == BY_CAM1_CAM3_TAG) {
        QMatrix4x4 cam3_to_cam1 = cam1_to_cam3.inverted();
        QMatrix4x4 tag_to_cam3  = cam3_to_tag.inverted();
        QMatrix4x4 tag_to_cam1  = tag_to_cam3 * cam3_to_cam1;
        QVector3D cam3_on_ground_tag(tag_to_cam3.column(3));
        QVector3D cam1_on_ground_tag(tag_to_cam1.column(3));
        QVector3D y_vector = cam3_on_ground_tag - cam1_on_ground_tag;
        y_vector.setZ(0);
        y_vector.normalize();
        QVector3D z_vector(0, 0, 1), x_vector;   // world in tag
        x_vector = QVector3D::crossProduct(y_vector, z_vector);
        x_vector.normalize();
        QMatrix4x4 tag_to_world, world_to_tag;
        tag_to_world.setToIdentity();
        tag_to_world.setColumn(0, x_vector);
        tag_to_world.setColumn(1, y_vector);
        tag_to_world.setColumn(2, z_vector);
        QVector4D cam3_pos_in_tag(tag_to_cam3.column(3));
        QVector4D world_pos_in_tag = cam3_pos_in_tag;
        world_pos_in_tag.setZ(0);
        tag_to_world.setColumn(3, world_pos_in_tag);
        world_to_tag             = tag_to_world.inverted();
        QMatrix4x4 world_to_cam1 = world_to_tag * tag_to_cam1;
        QVector3D view_ori(world_to_cam1.column(3));
        QVector3D view_dir(world_to_cam1.column(2));
        view_dir.normalize();
        QVector3D view_tar = view_ori + view_dir;
        assistant_transform_info.view.setToIdentity();
        assistant_transform_info.view.lookAt(view_ori, view_tar,
                                             QVector3D(0, 0, 1));
    } else if (update_assistant_transform_info_by == BY_CAM1_TAG) {
        QMatrix4x4 tag_to_cam1     = cam1_to_tag.inverted();
        QVector4D world_pos_in_tag = tag_to_cam1.column(3);
        world_pos_in_tag.setZ(0);
        QVector3D x, z(0, 0, 1);   // world in tag
        QVector3D y(tag_to_cam1.column(2));
        y.setZ(0);
        y.normalize();
        x = QVector3D::crossProduct(y, z);
        x.normalize();
        QMatrix4x4 tag_to_world;
        tag_to_world.setToIdentity();
        tag_to_world.setColumn(0, x);
        tag_to_world.setColumn(1, y);
        tag_to_world.setColumn(2, z);
        tag_to_world.setColumn(3, world_pos_in_tag);

        QMatrix4x4 world_to_tag  = tag_to_world.inverted();
        QMatrix4x4 world_to_cam1 = world_to_tag * tag_to_cam1;
        QVector3D view_ori(world_to_cam1.column(3));
        QVector3D view_dir(world_to_cam1.column(2));

        view_dir.normalize();
        QVector3D view_tar = view_ori + view_dir;
        assistant_transform_info.view.setToIdentity();
        assistant_transform_info.view.lookAt(view_ori, view_tar,
                                             QVector3D(0, 0, 1));
    }
    assistant_inited = true;
    glCheckError();
}

/**
 * @brief 从给定的主题名称和消息中获取交互式图像小部件的车道线数据
 *
 * 此回调函数用于解析传入的车道线信息，包括前轮驱动线、刀具操作线、
 * 刀具最大操作线以及设置零高度的刀具输入。
 *
 * @param _topic_name 主题名称，指明消息来源的主题。
 * @param msg 包含车道线数据的协议缓冲区消息对象。
 * @param _time 消息的时间戳。
 * @param _clock 消息的时钟时间戳。
 * @param _id 消息的ID，用于区分不同的消息实例。
 */
void InteractiveImageWidget::ecalLaneLineGetCallBack(
    const char* _topic_name, const xcmg_proto::lane_line& msg, long long _time,
    long long _clock, long long _id) {
    // 从消息中提取车道线数据
    front_wheel_driving_line    = msg.front_wheel_driving_line();
    blade_operation_line        = msg.blade_operation_line();
    blade_max_operation_line    = msg.blade_max_operation_line();
    blade_input_set_zero_height = msg.set_zero_height();
}

/**
 * 设置驾驶辅助系统的参数
 * 本函数用于设置驾驶辅助系统所使用的参数，包括车轮宽度距离、车轮高度距离、交叉间隔、轨迹长度和轨迹宽度。
 *
 * @param wheel_w_dis_ 车轮宽度距离，单位为米。
 * @param wheel_h_dis_ 车轮高度距离，单位为米。
 * @param cross_interval_ 交叉间隔，单位为米。
 * @param track_length_ 轨迹长度，单位为米。
 * @param track_width_ 轨迹宽度，单位为米。
 *
 * 本函数无返回值，设置完成后通过调用drive_assistant对象的相应函数完成参数的更新。
 */
void InteractiveImageWidget::setDriveAssistant(float wheel_w_dis_,
                                               float wheel_h_dis_,
                                               float cross_interval_,
                                               float track_length_,
                                               float track_width_) {
    // 调用drive_assistant对象的setDriveAssistant函数，传入参数完成设置
    drive_assistant->setDriveAssistant(wheel_w_dis_, wheel_h_dis_,
                                       cross_interval_, track_length_,
                                       track_width_);
}

void InteractiveImageWidget::setCameraGround(const cv::Mat& mat_cam3_to_tag_,
                                             const cv::Mat& mat_cam1_to_cam3_) {
    memcpy(cam3_to_tag.data(), mat_cam3_to_tag_.data, sizeof(float) * 16);
    cam3_to_tag = cam3_to_tag.transposed();
    memcpy(cam1_to_cam3.data(), mat_cam1_to_cam3_.data, sizeof(float) * 16);
    cam1_to_cam3                       = cam1_to_cam3.transposed();
    this->assistant_inited             = false;
    update_assistant_transform_info_by = BY_CAM1_CAM3_TAG;
}

/**
 * 设置相机到地面的变换矩阵
 * @param mat_cam1_to_tag
 * 一个4x4的矩阵，表示从相机1坐标系到标签（或地面）坐标系的变换关系。
 *                        该矩阵应为cv::Mat类型，数据类型为float。
 */
void InteractiveImageWidget::setCameraGround(const cv::Mat& mat_cam1_to_tag) {
    // 将传入的矩阵数据复制到类成员变量cam1_to_tag中
    memcpy(cam1_to_tag.data(), mat_cam1_to_tag.data, sizeof(float) * 16);
    // 由于原矩阵可能是非转置的，这里确保cam1_to_tag是转置后的状态
    cam1_to_tag = cam1_to_tag.transposed();
    // 重置辅助信息初始化状态，表示需要根据新的相机位置更新辅助信息
    this->assistant_inited = false;
    // 标记辅助信息更新来源为相机1到标签的变换
    update_assistant_transform_info_by = BY_CAM1_TAG;
}

void InteractiveImageWidget::setTrackCross(float width, float loc,
                                           float interval, float len) {
    drive_assistant->setTrackCross(width, loc, interval, len);
}

void InteractiveImageWidget::adjustTrackDelta(float delta_x, float delta_y,
                                              float delta_factor) {
    drive_assistant->adjustTrackDelta(delta_x, delta_y, delta_factor);
}

void InteractiveImageWidget::saveDriveAssistantTrackDelta() {
    drive_assistant->saveTrackDelta();
}

/**
 * 设置铲刀当前姿态
 * 该方法用于更新铲刀模型在可视化中的姿态，根据提供的六个参数计算并应用新的变换。
 *
 * @param l_CG CG距离，从铲刀中心到其重力中心的距离。
 * @param l_BD BD距离，从铲刀重力中心到其前端的距离。
 * @param l_EF EF距离，从铲刀前端到其尖端的距离。
 * @param theta 铲刀绕其前端与水平面的夹角。
 * @param l_IJ IJ距离，从铲刀尖端到其后端的距离。
 * @param l_LM LM距离，从铲刀后端到其尾部的距离。
 *
 * @note 该方法不返回任何值，它直接更新了铲刀模型的当前姿态。
 */
void InteractiveImageWidget::setBladeCurrentPose(float l_CG, float l_BD,
                                                 float l_EF, float theta,
                                                 float l_IJ, float l_LM) {
    // 将提供的参数用于更新铲刀模型的当前姿态
    drive_assistant->blade_model_transformer->setCurrentPoseNew(
        l_CG, l_BD, l_EF, theta, l_IJ, l_LM);
}

/**
 * 获取跟踪帧缓冲区在CUDA中的指针
 * 
 * 本函数用于获取当前对象中存储的跟踪帧缓冲区（frame buffer）对应的CUDA内存指针。
 * 这个指针可以用于在CUDA计算环境中直接访问和操作该帧缓冲区的数据。
 * 
 * @return 返回一个float类型的指针，指向在CUDA设备内存中的跟踪帧缓冲区数据。
 */
float* InteractiveImageWidget::getTrackFBCudaPtr() {
    return fb_track_->getCudaPtr(); // 从fb_track_对象中获取CUDA内存指针
}

/**
 * 更新在Cuda内存中的跟踪信息。
 * 
 * 该函数调用fb_track_对象的updateCudaMem()方法，以更新在Cuda内存中的跟踪信息。
 * 
 * @return 返回一个布尔值，表示是否成功更新了Cuda内存中的跟踪信息。
 */
bool InteractiveImageWidget::updateTrackInCudaMem() {
    return fb_track_->updateCudaMem();
}

/**
 * 获取铲刀模型的最低Z值
 *
 * 本函数用于从铲刀模型转换器中获取当前铲刀模型中Z轴方向上的最低值。
 * 这对于交互式图像的处理和显示，特别是在进行3D铲刀切割操作时，
 * 是非常关键的。该值可以用于确定铲刀在图像中的深度位置。
 *
 * @return 返回铲刀模型转换器中铲刀模型的最低Z值。
 */
int InteractiveImageWidget::getBladeLowestZ() {
    // 通过驱动助手获取铲刀模型转换器，并从中得到最低的Z值
    return drive_assistant->blade_model_transformer->getLowestZ();
}
