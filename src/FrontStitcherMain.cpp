#include "FrontStitcherMain.h"
#include "common/EcalImageSender.h"
#include "common/GstReceiver.h"
#include "common/cylinder_stitcher.cuh"
#include "common/cylinder_stitcher.h"
#include "stage/ThreadInteractiveWidget.h"
#include "stage/interactiveimagewidget.h"
#include "util/innoreal_timer.hpp"
#include <Eigen/Eigen>
#include <QApplication>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <stack>

// #define TEST_YOLO

#ifdef TEST_YOLO
#include "yolo_detect.h"
#endif

#define USE_VIDEO_INPUT

int frontStitcherMain(int argc, char** argv) {
#ifdef TEST_YOLO
    yolo_detect::YoloDetect yolo_detector("../assets/weights/best.onnx", true);
#endif

    // 实例化一个ThreadInteractiveWidget对象
    ThreadInteractiveWidget thread_interactive_widget;
    float crop_factor = .0;   // 裁剪因子，用于调整内参矩阵的焦点。
    int view_num = 3, src_image_width = 1920, src_image_height = 1080;
#ifdef USE_INPUT_RESIZE_FACTOR
    float input_resize_factor = 0.85;
    src_image_height *= input_resize_factor;
    src_image_width *= input_resize_factor;
#endif

    int cropped_img_width =
        src_image_width - int(src_image_width * crop_factor) * 2;

    std::vector<int> camera_name_map = {9, 5, 8};   // 左、中、右
    int mid_camera_name              = camera_name_map[1];

    std::vector<std::vector<int>> yawpitchroll = {
        {-22, 0, 0}, {0, 0, 0}, {22, 0, 0}};

    std::vector<float4> intrin_vec;    //存储相机内参的向量
    std::vector<float4> distort_vec;   //存储相机畸变参数的向量
    std::vector<float> extrin_vec;     //存储相机外参的向量
    Eigen::Matrix4f
        ref_camera_pose;   // 参考相机的姿态（旋转和平移），选择中间视图作为参考

    std::filesystem::path yamls_dir("../assets/yamls");
    std::filesystem::path video_dir("../assets/test_video2");

    for (int camera_idx = 0; camera_idx < view_num; ++camera_idx) {
        Eigen::Matrix3f K;
        Eigen::Vector4f D;
        // 读取相机内参
        std::filesystem::path fs_path("../assets/yamls");
        fs_path.append("camera_" + std::to_string(camera_name_map[camera_idx]) +
                       "_intrin.yaml");
        cv::FileStorage fs;
        if (!fs.open(fs_path, cv::FileStorage::READ)) {
            printf("cannot open %s\n", fs_path.c_str());
        }
        cv::Mat K_mat;
        fs["K"] >> K_mat;
        K_mat.at<double>(0, 2) -= src_image_width * crop_factor;
#ifdef USE_INPUT_RESIZE_FACTOR
        K_mat *= input_resize_factor;
        K_mat.at<double>(2, 2) = 1.0;
#endif
        // 从文件存储中读取D矩阵
        cv::Mat D_mat;
        fs["D"] >> D_mat;   // 从文件存储fs中读取名为"D"的数据到D_mat
        fs.release();       // 释放文件存储fs的资源

        // 将OpenCV的矩阵格式转换为Eigen的矩阵格式
        cv::cv2eigen(K_mat, K);   // 将OpenCV的K_mat矩阵转换为Eigen的K矩阵
        cv::cv2eigen(D_mat, D);   // 将OpenCV的D_mat矩阵转换为Eigen的D矩阵

        Eigen::Matrix4f extrin;   // 相机外参矩阵
        // 计算相机外参，如果是第一个相机，则直接使用单位矩阵，否则从文件中读取相对变换矩阵

        // 将yawpitchroll数组中的角度转换为弧度，并初始化eulerAngle1
        Eigen::Vector3f eulerAngle1(
            yawpitchroll[camera_idx][0] * 3.141592653 / 180.0,
            yawpitchroll[camera_idx][1] * 3.141592653 / 180.0,
            yawpitchroll[camera_idx][2] * 3.141592653 / 180.0);

        // 分别为yaw, pitch, roll角度初始化AngleAxis对象
        Eigen::AngleAxisf yawAngle1(
            Eigen::AngleAxisf(eulerAngle1(0), Eigen::Vector3f::UnitY()));
        Eigen::AngleAxisf pitchAngle1(
            Eigen::AngleAxisf(eulerAngle1(1), Eigen::Vector3f::UnitX()));
        Eigen::AngleAxisf rollAngle2(
            Eigen::AngleAxisf(eulerAngle1(2), Eigen::Vector3f::UnitZ()));

        // 组合三个旋转角得到最终的旋转向量
        Eigen::AngleAxisf rotation_vector;
        rotation_vector = yawAngle1 * pitchAngle1 * rollAngle2;

        // 初始化变换矩阵T，并设置其旋转部分为旋转向量的旋转矩阵
        Eigen::Matrix4f T     = Eigen::Matrix4f::Identity();
        T.topLeftCorner(3, 3) = rotation_vector.toRotationMatrix();

        // 计算相机的外参矩阵，若是相机索引为1，则直接设为单位矩阵
        extrin = T.inverse();

        if (camera_idx == 1) {
            extrin = Eigen::Matrix4f::Identity();
        } else {
            // 根据相机索引，从yaml文件中读取相对变换矩阵，并转换为Eigen矩阵格式
            int camera_name = camera_name_map[camera_idx];
            std::string fs_file_name =
                "camera_extrin_" + std::to_string(mid_camera_name) + "_and_" +
                std::to_string(camera_name) + ".yaml";
            std::filesystem::path fs_file_path("../assets/yamls");
            fs_file_path.append(fs_file_name);
            cv::FileStorage fs;
            if (!fs.open(fs_file_path, cv::FileStorage::Mode::READ)) {
                printf("cannot open fs %s\n", fs_file_path.c_str());
                exit(0);
            }
            cv::Mat rel_matrix;   // middle camera to i camera
            std::string rel_matrix_name = "matrix" +
                                          std::to_string(mid_camera_name) +
                                          "to" + std::to_string(camera_name);
            fs[rel_matrix_name] >> rel_matrix;
            rel_matrix = rel_matrix.inv();
            cv::cv2eigen(rel_matrix, extrin);
            extrin.col(3) = Eigen::Vector4f(0, 0, 0, 1);
        }

        // 添加相机内参和畸变参数到向量中
        intrin_vec.emplace_back(
            make_float4(K(0, 0), K(1, 1), K(0, 2), K(1, 2)));
        distort_vec.emplace_back(make_float4(D(0), D(1), D(2), D(3)));

        // 添加相机外参到向量中
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                extrin_vec.emplace_back(extrin(row, col));
            }
        }

        // 选择中间视图作为参考相机姿态
        if (camera_idx == (view_num / 2)) {
            ref_camera_pose = extrin.inverse();
        }
    }

#ifdef USE_INPUT_RESIZE_FACTOR
    int ar_vis_width = 1920, ar_vis_height = 1080;
    ar_vis_width *= input_resize_factor;
    ar_vis_height *= input_resize_factor;
#endif
    // 初始化相机位姿矩阵
    Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
    camera_pose(2, 3)           = -10;

    // 读取并设置畸变矫正后的内参
    cv::FileStorage fs_undistort_intrin;
    std::filesystem::path undistort_intrin_path(yamls_dir);
    undistort_intrin_path.append("camera_" + std::to_string(mid_camera_name) +
                                 "_intrin_undistort.yaml");
    // 打开并读取内参文件，若失败则打印错误信息
    if (!fs_undistort_intrin.open(undistort_intrin_path,
                                  cv::FileStorage::READ)) {
        printf("cannot open %s \n", undistort_intrin_path.c_str());
    }
    cv::Mat undistorted_K;
    fs_undistort_intrin["K"] >> undistorted_K;
    // 调整内参矩阵的中心点
    undistorted_K.at<double>(0, 2) -= src_image_height * crop_factor;

    // 从内参矩阵中提取出相关参数
    float fx = undistorted_K.at<double>(0, 0),
          fy = undistorted_K.at<double>(1, 1);
    float cx = undistorted_K.at<double>(0, 2),
          cy = undistorted_K.at<double>(1, 2);

    // 设置初始的俯仰翻滚角
    float yaw = 0 * 3.14159 / 180, pitch = 0 * 3.14159 / 180,
          roll = 80 * 3.14159 / 180;
    Eigen::Vector3f eulerAngle2(yaw, pitch, roll);
    // 通过欧拉角构建旋转矩阵
    Eigen::AngleAxisf rollAngle2(
        Eigen::AngleAxisf(eulerAngle2(2), Eigen::Vector3f::UnitX()));
    Eigen::AngleAxisf pitchAngle2(
        Eigen::AngleAxisf(eulerAngle2(1), Eigen::Vector3f::UnitY()));
    Eigen::AngleAxisf yawAngle2(
        Eigen::AngleAxisf(eulerAngle2(0), Eigen::Vector3f::UnitZ()));
    Eigen::AngleAxisf rotation_vector;
    rotation_vector = yawAngle2 * pitchAngle2 * rollAngle2;

    // 将旋转矩阵部分赋值给相机位姿矩阵
    camera_pose.topLeftCorner(3, 3) = rotation_vector.toRotationMatrix();
    // 准备额外的内参和外参数据
    float4 extra_intrin;
    std::vector<float> extra_extrin_vec(16);
    float* extrin = extrin_vec.data() + view_num / 2 * 16;

    // 设置额外的内参
    extra_intrin = make_float4(fx, fy, cx, cy);
    // 复制外参数据
    memcpy(extra_extrin_vec.data(), extrin, 16 * sizeof(float));

#ifdef USE_VIDEO_INPUT
    // 使用视频数据
    std::vector<cv::VideoCapture> video_captures(view_num);
    for (int i = 0; i < view_num; i++) {
        std::filesystem::path video_path(video_dir);
        video_path.append("camera_video_" + std::to_string(camera_name_map[i]) +
                          ".avi");
        if (!video_captures[i].open(video_path)) {
            printf("cannot open %s\n", video_path.c_str());
        }
    }
#else
    // 使用实时gst码流
    std::vector<std::string> gst_strs = {
        "udpsrc port=5001 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
        "udpsrc port=5002 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
        "udpsrc port=5000 caps=application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96 ! rtpjitterbuffer ! rtph264depay ! h264parse ! queue ! avdec_h264 ! videoconvert ! video/x-raw ! appsink sync=false",
    };

    /**
     * 初始化并启动多个GstReceiver实例。
     *
     * @param view_num GstReceiver实例的数量。
     *
     * 此函数首先初始化指定数量的GstReceiver实例，然后尝试启动它们的接收工作线程。
     * 对于每个实例，如果初始化成功，将打印一条相应的消息。
     */
    std::vector<GstReceiver> gst_receivers(view_num);
    for (int i = 0; i < view_num; i++) {
        printf("initialize VideoCapture %d\n", i);
        // 尝试初始化GstReceiver实例
        if (gst_receivers[i].initialize(gst_strs[i], 2)) {
            printf("initialize VideoCapture %d done\n", i);
        }
    }
    // 启动所有成功初始化的GstReceiver实例的工作线程
    for (int i = 0; i < view_num; i++) {
        if (gst_receivers[i].startReceiveWorker()) {
            printf("start gst_receiver %d done\n", i);
        }
    }
#endif

    // 初始化CylinderStitcherGPU对象
    CylinderStitcherGPU cylinder_stitcher(
        view_num, cropped_img_width * src_image_height * view_num * 1.5);
    cylinder_stitcher.novel_images_num_ = 5;   // 设置新视图的数量
    // 设置相机参数
    cylinder_stitcher.setCameras(intrin_vec, distort_vec, extrin_vec);
    // 设置额外视图的相机参数
    cylinder_stitcher.setExtraViewCamera(extra_intrin, extra_extrin_vec);
    // 创建掩膜矢量
    std::vector<uchar> masks(cropped_img_width * src_image_height * view_num);
    // 创建图像矢量
    std::vector<uchar3> images(cropped_img_width * src_image_height * view_num);
    // 为每个视图创建并处理掩膜图像，然后将其复制到掩膜矢量中
    for (int view_idx = 0; view_idx < view_num; ++view_idx) {
        // 创建全白的掩膜图像
        cv::Mat mask_image =
            cv::Mat::ones(src_image_height, cropped_img_width, CV_8UC1) * 255;
        // 获取用于侵蚀操作的结构元素
        cv::Mat erode_kernel =
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        // 对掩膜图像进行侵蚀处理以减小噪声
        cv::erode(mask_image, mask_image, erode_kernel);
        // 将处理后的掩膜图像数据复制到掩膜矢量中
        memcpy(masks.data() + view_idx * src_image_height * cropped_img_width,
               mask_image.data, src_image_height * cropped_img_width);
    }

    // 设置遮罩掩码
    cylinder_stitcher.setMasks(
        masks.data(), cropped_img_width, src_image_height,
        view_num);   // TODO CHECK: 需要检查这一行的功能是否正确实现

    // 计算视点原点位置
    float3 origin = make_float3(ref_camera_pose.topRightCorner(3, 1)(0, 0),
                                ref_camera_pose.topRightCorner(3, 1)(1, 0),
                                ref_camera_pose.topRightCorner(3, 1)(2, 0));
    // 调整原点位置，偏移量为800，调整方式基于相机矩阵中的第三个列向量
    origin =
        origin - 800 * make_float3(ref_camera_pose.topLeftCorner(3, 3)(0, 2),
                                   ref_camera_pose.topLeftCorner(3, 3)(1, 2),
                                   ref_camera_pose.topLeftCorner(3, 3)(2, 2));
    // 更新视点原点位置
    cylinder_stitcher.novel_view_pos_ = origin;

    // 启动主线程工作线程
    thread_interactive_widget.startMainWorkerThread(argc, argv);
    // 等待直到有新的图像数据可用
    while (!thread_interactive_widget.checkHaveNewImage())
        ;

    // 初始化EcalImageSender对象，并打开"front_switcher"通道
    EcalImageSender ecal_image_sender;
    ecal_image_sender.open("front_switcher");

    // 初始化变量
    int time        = 0;                    // 用于记录时间
    bool video_done = false;                // 标记视频是否处理完成
    innoreal::InnoRealTimer timer;          // 初始化计时器
    std::vector<cv::Mat> image(view_num);   // 初始化存储图像的向量

    // 初始化时间窗口参数
    int window_size = 30;            // 时间窗口大小
    std::queue<float> time_window;   // 用于存储时间戳的时间窗口
    float time_sum = 0;   // 用于计算时间窗口内时间戳的总和

    while (true) {
        timer.TimeStart();
        // 遍历所有视图索引，进行图像捕获和处理
        for (int view_idx = 0; view_idx < view_num; ++view_idx) {
#ifdef USE_VIDEO_INPUT
            // 尝试从视频中读取一帧图像，如果视频播放结束，则重新从头开始
            video_done = !(video_captures[view_idx].read(image[view_idx]));
            if (video_done) {
                video_captures[view_idx].set(cv::CAP_PROP_POS_FRAMES, 0);
                // 重试读取图像
                video_done = !(video_captures[view_idx].read(image[view_idx]));
            }
#ifdef USE_INPUT_RESIZE_FACTOR
            // 如果定义了输入缩放因子，则调整图像尺寸
            cv::resize(image[view_idx], image[view_idx],
                       cv::Size(src_image_width, src_image_height));
#endif
#else   // USE_VIDEO_INPUT
        // 如果没有使用视频输入，则从GStreamer接收器获取图像
            image[view_idx] = gst_receivers[view_idx].getImageMat();
#endif

#ifdef USE_720P_TO_1080P
            // 如果定义了将720P转换为1080P，则调整图像尺寸
            cv::resize(image[view_idx], image[view_idx], cv::Size(1920, 1080));
#endif

            // 将处理后的图像数据复制到统一的图像数据数组中
            memcpy((uchar*)(images.data() +
                            view_idx * src_image_height * cropped_img_width),
                   image[view_idx].data,
                   src_image_height * cropped_img_width * sizeof(uchar3));
        }

        // 释放所有图像视图的资源
        for (int i = 0; i < view_num; i++) {
            image[i].release();   // 释放第i个图像视图的资源
        }
        // 设置图像数据供 cylinder_stitcher 使用
        cylinder_stitcher.setImages(images.data(), cropped_img_width,
                                    src_image_height, view_num);
        // 执行图像到圆柱面的映射
        cylinder_stitcher.stitch_project_to_cyn(time);
        // 执行对齐和缝合，并进行遮罩融合
        cylinder_stitcher.stitch_align_seam_blend(time);

        // 创建输出图像和掩码的矩阵
        cv::Mat out_image, out_mask;

        // 设置额外的图像数据（来自线程交互部件），用于 CUDA 处理
        cylinder_stitcher.setExtraImageCuda(
            thread_interactive_widget.getTrackImageCudaPtr(), 1920, 1080);

        // 将当前项目缝合到屏幕，并获取最终的CPU图像和掩码
        cylinder_stitcher.stitch_project_to_screen(time);
        cylinder_stitcher.getFinalImageCPU(out_image, out_mask);

        // 更新时间戳，并将图像上部70像素复制到图像的下部
        ++time;
        out_image(cv::Rect(0, 70, out_image.cols, out_image.rows - 80))
            .copyTo(out_image);
        cv::resize(out_image, out_image, cv::Size(1280, 720));
        // 发送调整后的图像
        ecal_image_sender.pubImage(out_image);

        // 结束计时，计算此帧处理的时间
        timer.TimeEnd();

        // 计算总时间并更新时间窗口，保持时间窗口大小不变
        time_sum += timer.TimeGap_in_ms();
        time_window.push(timer.TimeGap_in_ms());
        while (time_window.size() > window_size) {
            time_sum -= time_window.front();
            time_window.pop();
        }
        // 计算时间窗口内的平均处理时间
        float mean_time = time_sum / time_window.size();
        // 打印平均时间和当前窗口内帧数的信息
        printf("mean time cost: %fms in %d frames\n", mean_time, window_size);

        // 打印当前帧的处理时间和总时间的信息
        printf(
            "=========================%d frame time: %f ms=========================\n",
            time, timer.TimeGap_in_ms());
#ifdef TEST_YOLO
        cv::Mat result = yolo_detector.detect(out_image);
        ecal_image_sender.pubImage(result);
#endif
    }
    return 0;
}
