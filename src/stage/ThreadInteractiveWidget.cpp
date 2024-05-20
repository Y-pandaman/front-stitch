#include "stage/ThreadInteractiveWidget.h"

/**
 * @brief 启动转向角Ecal接收器
 *
 * 该函数用于初始化并启动一个用于接收转向角数据的Ecal订阅者。
 *
 * @param topic_name 订阅的主题名称，类型为std::string。
 * @param queue_size_ 队列大小，类型为int，决定接收缓存的大小。
 * @return 总是返回true，表示启动成功。
 */
bool ThreadInteractiveWidget::startSteeringAngleEcalReceiver(
    const std::string& topic_name, int queue_size_) {
    // 检查eCAL是否已初始化，若未初始化则进行初始化
    if (!eCAL::IsInitialized())
        eCAL::Initialize();
    // 启用环回，以便可以在本地发布和订阅相同的话题
    eCAL::Util::EnableLoopback(true);

    // 创建一个指向xcmg_proto::Grader_CAN类型的protobuf订阅者
    m_sub_steering_angle =
        std::make_shared<eCAL::protobuf::CSubscriber<xcmg_proto::Grader_CAN>>(
            topic_name);

    // 绑定接收回调函数
    auto cb = std::bind(&ThreadInteractiveWidget::ecalSteeringAngleCallBack,
                        this, std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4,
                        std::placeholders::_5);

    // 添加接收回调，以便在接收到消息时调用
    m_sub_steering_angle->AddReceiveCallback(cb);
    return true;
}

/**
 * @brief
 * 回调函数，用于处理接收到的挖掘机CAN总线数据，计算并更新铲斗的倾斜角度。
 *
 * @param _topic_name 话题的名称，用于标识接收到的数据的话题。
 * @param msg 包含从CAN总线接收到的消息数据的协议缓冲区对象。
 * @param _time 消息的时间戳。
 * @param _clock 接收消息时的系统时钟。
 * @param _id 消息的ID号。
 */
void ThreadInteractiveWidget::ecalSteeringAngleCallBack(
    const char* _topic_name, const xcmg_proto::Grader_CAN& msg,
    long long int _time, long long int _clock, long long int _id) {
    // 获取油门信号值
    float input = msg.throttle_signal();
    // 定义输入信号的最小值、最大值及中间值
    float input_min = 3139, input_max = 14368;
    float input_mid = 9300;
    // 定义角度的最小值、最大值及中间值
    float angle_min = -44, angle_max = 44;
    float angle_mid = 0;
    // 根据输入信号计算对应的倾斜角度
    float angle = 0;
    if (input <= input_mid) {
        angle = (input - input_min) / (input_mid - input_min) *
                    (angle_mid - angle_min) +
                angle_min;
    } else {
        angle = (input - input_mid) / (input_max - input_mid) *
                    (angle_max - angle_mid) +
                angle_mid;
    }
    // 更新当前角度值，并标记为已更新
    current_angle        = -angle / 180 * 3.1415926;
    current_angle_update = true;

    // 更新铲刀相关参数
    l_cg        = (msg.blades_left()) + 780;     // 左铲刀位置
    l_bd        = (msg.blades_right()) + 780;    // 右铲刀位置
    l_ef        = (msg.blades_slante()) + 970;   // 斜铲刀位置
    blade_theta = -(msg.turn_circle() - 65);     // 铲斗旋转角度修正
    blade_theta = -(msg.turn_circle() -
                    45);   // 铲斗旋转角度再次修正（此处可能有误，应检查逻辑）
    l_IJ        = (msg.shovel_corner_right()) + 500;   // 右侧铲斗角位置
    l_LM        = (msg.blades_translate()) + 2110;     // 铲斗平移位置
}

/**
 * 主工作线程函数，负责初始化和持续更新交互式图像小部件。
 *
 * @param argc 传入的命令行参数个数。
 * @param argv 传入的命令行参数值。
 */
void ThreadInteractiveWidget::mainWorker(int argc, char** argv) {
    // 初始化图像发送者
    blade_image_sender.open("Blade_MR");
    back_track_image_sender.open("Back_Track_Image");
    blade_height_sender.open("Blade_Relative_Height_Topic");

    // 初始化相机和平面的位姿
    Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f plane_pose  = Eigen::Matrix4f::Identity();
    std::filesystem::path yamls_dir("../assets/yamls");
    // 读取相机内参
    cv::FileStorage fs_undistort_intrin;
    std::filesystem::path undistort_intrin_path(yamls_dir);
    undistort_intrin_path.append(
        "camera_5_intrin_undistort.yaml");   // 中间相机的内参
    if (!fs_undistort_intrin.open(undistort_intrin_path,
                                  cv::FileStorage::READ)) {
        printf("cannot open %s \n", undistort_intrin_path.c_str());
    }
    cv::Mat undistorted_K;
    fs_undistort_intrin["K"] >> undistorted_K;
    // 解析相机参数
    int ar_vis_width = 1920, ar_vis_height = 1080;
    float fx = undistorted_K.at<double>(0, 0),
          fy = undistorted_K.at<double>(1, 1);
    float cx = undistorted_K.at<double>(0, 2),
          cy = undistorted_K.at<double>(1, 2);

    // 设置初始俯仰角和偏航角
    float yaw = 0, pitch = 0, roll = 50 * 3.14159 / 180;
    Eigen::Vector3f eulerAngle(yaw, pitch, roll);
    // 构建旋转矩阵
    Eigen::AngleAxisf rollAngle(
        Eigen::AngleAxisf(eulerAngle(2), Eigen::Vector3f::UnitX()));
    Eigen::AngleAxisf pitchAngle(
        Eigen::AngleAxisf(eulerAngle(1), Eigen::Vector3f::UnitY()));
    Eigen::AngleAxisf yawAngle(
        Eigen::AngleAxisf(eulerAngle(0), Eigen::Vector3f::UnitZ()));
    Eigen::AngleAxisf rotation_vector;
    rotation_vector                 = yawAngle * pitchAngle * rollAngle;
    camera_pose.topLeftCorner(3, 3) = rotation_vector.toRotationMatrix();
    camera_pose(2, 3)               = -10;

    // 初始化交互式图像小部件,订阅车道线显示开关Lane_Line_msg
    cv::FileStorage fs;
    QApplication a(argc, argv);
    InteractiveImageWidget w(ar_vis_width, ar_vis_height, fx, fy, cx, cy,
                             camera_pose, plane_pose, nullptr, 1920, 1080);

    widget_ptr = &w;

    // 显示窗口并设置窗口状态
    w.show();
    w.setWindowState(Qt::WindowMinimized);

    // 设置驱动辅助参数
    float wheel_w_dis_ = 3.506, wheel_h_dis_ = 6.405, cross_interval_ = 5,
          track_length_ = 30, track_width_ = 0.02;   // 2605
    w.setDriveAssistant(wheel_w_dis_, wheel_h_dis_, cross_interval_,
                        track_length_, track_width_);
    w.setWheelTheta(0);
    // 加载地面矩阵配置
    std::string filepath("../assets/yamls/ground.yaml");
    if (!fs.open(filepath, cv::FileStorage::READ)) {
        printf("cannot open %s\n", filepath.c_str());
        return;
    }
    cv::Mat mat_cam1_to_tag;
    fs["ground_matrix"] >> mat_cam1_to_tag;
    fs.release();

    w.setCameraGround(mat_cam1_to_tag);

    // 初始化铲刀最大操作线、操作线和前轮驱动线，并绘制铲刀模型
    w.setBladeMaxOperationLine(true);
    w.setBladeOperationLine(true);
    w.setFrontWheelDrivingLine(true);
    w.setDrawBladeModel(true);
    w.update();

    have_new_image = true;
#ifdef OUTPUT_BLADE_VIDEO
    // 初始化铲刀视频写入器
    cv::VideoWriter blade_video_writer;
    blade_video_writer.open("./output/blade_video.avi",
                            cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
                            cv::Size(1920, 1080), true);
#endif

    // 无限循环，更新铲刀位姿和图像
    for (int i = 0;; i++) {
        float angle = current_angle;
        w.setWheelTheta(angle);
        // 设置当前铲刀位姿
        float cg_m = (float)l_cg / 1000.0f, bd_m = (float)l_bd / 1000.0f,
              ef_m = (float)l_ef / 1000.0f;
        float ij_m = (float)l_IJ / 1000.0f, lm_m = (float)l_LM / 1000.0f;
        float theta_radian = (float)blade_theta / 180.0f * 3.1415926f;

        w.setBladeCurrentPose(cg_m, bd_m, ef_m, theta_radian, ij_m, lm_m);
        // 获取铲刀最低Z值并更新图像
        float lowest_z = w.getBladeLowestZ();
        w.update();
        // 获取并发送铲刀图像
        cv::Mat blade_image;
        w.getImageCPU4Channels(blade_image, InteractiveImageWidget::FB_BLADE);
        blade_image_sender.pubImage(blade_image);

        // 获取并发送后视轨迹图像
        cv::Mat back_track_image;
        w.getImageCPU(back_track_image, InteractiveImageWidget::FB_BACK_TRACK);

        cv::flip(back_track_image, back_track_image, 1);
        back_track_image_sender.pubImage(back_track_image);

        // 获取并发送左右铲刀高度信息
        float left_height, right_height;
        w.drive_assistant->blade_model_transformer->getRelativeHeight(
            left_height, right_height);
        blade_height_sender.pubMessage(left_height, right_height);

#ifdef OUTPUT_BACK_TRACK_IMAGE
        // 输出后视图像
        cv::imwrite("./output/image/back_track_" + std::to_string(i) + ".png",
                    back_track_image);
        cv::Mat track_image;
        w.getImageCPU(track_image, InteractiveImageWidget::FB_TRACK);
        cv::imwrite("./output/image/track_" + std::to_string(i) + ".png",
                    track_image);
#endif

#ifdef OUTPUT_BLADE_IMAGE
        // 输出铲刀图像
        cv::imwrite("./output/image/blade_" + std::to_string(i) + ".png",
                    blade_image);
#endif

#ifdef OUTPUT_BLADE_VIDEO
        // 输出铲刀视频帧
        cv::Mat blade_image_3channels;
        w.getImageCPU(blade_image_3channels, InteractiveImageWidget::FB_BLADE);
        blade_video_writer.write(blade_image_3channels);
#endif

        // 更新轨迹图像内存
        w.updateTrackInCudaMem();
        cv::waitKey(1);

        have_new_image       = true;
        current_angle_update = false;
    }
}

/**
 * 启动主工作线程。
 * 本函数用于创建并启动一个名为main_worker_thread的线程，该线程将执行mainWorker函数。
 * 如果主线程已存在（即之前已被启动），则不会重新启动，并返回false。
 *
 * @param argc 传递给应用程序的参数个数。
 * @param argv 传递给应用程序的参数值数组。
 * @return
 * 返回一个布尔值，表示是否成功启动了主工作线程。成功返回true，失败返回false。
 */
bool ThreadInteractiveWidget::startMainWorkerThread(int argc, char** argv) {
    // 检查主工作线程是否已存在，若存在则不启动，直接返回false
    if (main_worker_thread != nullptr) {
        return false;
    }

    // 启动方向盘角度校准接收器
    startSteeringAngleEcalReceiver("Grader_CAN_Status", 2);
    // 创建并启动主工作线程，线程执行函数为mainWorker，传入参数为this及argc,
    // argv
    main_worker_thread =
        new std::thread(&ThreadInteractiveWidget::mainWorker, this, argc, argv);
    return true;
}

/**
 * 获取跟踪图像的CUDA指针
 * 
 * 该函数是一个成员函数，属于ThreadInteractiveWidget类。它通过调用widget_ptr成员变量的
 * getTrackFBCudaPtr()函数来获取一个指向跟踪图像帧的CUDA内存指针。这个功能通常用于在CUDA
 * 环境中对跟踪图像进行处理。
 * 
 * @return 返回一个float类型的指针，该指针指向CUDA内存中的跟踪图像数据。
 */
float* ThreadInteractiveWidget::getTrackImageCudaPtr() {
    return widget_ptr->getTrackFBCudaPtr(); // 从widget_ptr中获取跟踪图像的CUDA指针
}
