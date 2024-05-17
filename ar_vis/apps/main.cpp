#include "blade_message.pb.h"
#include "interactiveimagewidget.h"
#include <Eigen/Eigen>
#include <QApplication>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

std::atomic_int l_cg = 1311, l_bd = 1328, l_ef = 1247, blade_theta = 0,
                l_nanb = 960;

void ecalBladeInputCallback(const char* _topic_name,
                            const BladeController::BladeInput& msg,
                            long long int _time, long long int _clock,
                            long long int _id) {
    // TODO translate
    l_cg        = msg.l_cg();
    l_bd        = msg.l_bd();
    l_ef        = msg.l_ef();
    blade_theta = msg.theta();
    l_nanb      = msg.l_nanb();
}

int main(int argc, char** argv) {
    int ar_vis_width = 1920, ar_vis_height = 1080;
    Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f plane_pose  = Eigen::Matrix4f::Identity();
    std::filesystem::path yamls_dir("../example/yamls");
    // set extra intrin
    cv::FileStorage fs_undistort_intrin;
    std::filesystem::path undistort_intrin_path(yamls_dir);
    undistort_intrin_path.append("camera_1_intrin_undistort.yaml");
    if (!fs_undistort_intrin.open(undistort_intrin_path,
                                  cv::FileStorage::READ)) {
        printf("cannot open %s \n", undistort_intrin_path.c_str());
    }
    cv::Mat undistorted_K;
    fs_undistort_intrin["K"] >> undistorted_K;
    float fx = 324.0f, fy = 324.0f, cx = (ar_vis_width - 1) * 0.5f,
          cy  = (ar_vis_height - 1) * 0.5f;
    float yaw = 0, pitch = 0, roll = 50 * 3.14159 / 180;
    Eigen::Vector3f eulerAngle(yaw, pitch, roll);
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

    cv::FileStorage fs;

    QApplication a(argc, argv);
    InteractiveImageWidget w(ar_vis_width, ar_vis_height, fx, fy, cx, cy,
                             camera_pose, plane_pose, 0, 0);
    w.show();

    // track parameters
    float wheel_w_dis_ = 3.5, wheel_h_dis_ = 6, cross_interval_ = 5,
          track_length_ = 60, track_width_ = 0.02;
    w.setDriveAssistant(wheel_w_dis_, wheel_h_dis_, cross_interval_,
                        track_length_, track_width_);
    w.setKeyBoardInput(true);
    std::string filepath("../example/yamls/ground.yaml");
    if (!fs.open(filepath, cv::FileStorage::READ)) {
        printf("cannot open %s\n", filepath.c_str());
        return 0;
    }
    cv::Mat mat_cam1_to_tag;
    fs["ground_matrix"] >> mat_cam1_to_tag;
    fs.release();

    // camera & ground parameters
    w.setCameraGround(mat_cam1_to_tag);
    printf("set cameraground done\n");

    w.setBladeMaxOperationLine(true);
    w.setBladeOperationLine(false);
    w.setFrontWheelDrivingLine(true);
    w.setDrawBladeModel(true);

    w.update();
    int num = 30;
    w.setWheelTheta(0);
    float min_blade_angle = -45, max_blade_angle = 45;
    int video_length     = 20;   // s
    int video_frame_rate = 25;
    int max_frame        = video_length * video_frame_rate;
    float cur_step  = (max_blade_angle - min_blade_angle) / (float)max_frame;
    float cur_angle = min_blade_angle;

    cv::VideoWriter video_writer;
    video_writer.open("./output/video/blade_video.avi",
                      cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 25,
                      cv::Size(ar_vis_width, ar_vis_height), true);

    if (!eCAL::IsInitialized()) {
        eCAL::Initialize();
        eCAL::Util::EnableLoopback(true);
    }
    std::shared_ptr<eCAL::protobuf::CSubscriber<BladeController::BladeInput>>
        sub_blade_input = std::make_shared<
            eCAL::protobuf::CSubscriber<BladeController::BladeInput>>(
            "Blade Input Topic");

    auto cb = std::bind(&ecalBladeInputCallback, std::placeholders::_1,
                        std::placeholders::_2, std::placeholders::_3,
                        std::placeholders::_4, std::placeholders::_5);

    sub_blade_input->AddReceiveCallback(cb);

    for (int i = 0;; i++) {
        printf("print %d times\n", i);
        float cg_m = (float)l_cg / 1000.0f, bd_m = (float)l_bd / 1000.0f,
              ef_m = (float)l_ef / 1000.0f, nanb_m = (float)l_nanb / 1000.0f;
        float theta_radian = (float)blade_theta / 180.0f * 3.1415926f;
        w.setBladeCurrentPose(cg_m, bd_m, ef_m, theta_radian, nanb_m, 0);

        w.setWheelTheta(3 / 180.0 * 3.14159);

        w.update();
        cv::Mat blade_image;
        w.getImageCPU(blade_image, InteractiveImageWidget::FB_BLADE);
        std::cout << "ar_image channels: " << blade_image.channels()
                  << std::endl;
        printf("ar_image width:%d, height:%d\n", blade_image.cols,
               blade_image.rows);

        cv::Mat ar_image;
        w.getImageCPU(ar_image, InteractiveImageWidget::FB_TRACK);

        cv::imshow("test", ar_image);
        static cv::Mat img2 = cv::imread(
            "/home/touch/data/1226_calib_ground_/1226_calib_ground_1/"
            "noblur_camera_video_1/1-0.png");

        cv::Mat result;
        cv::addWeighted(ar_image, 0.5, img2, 0.5, 0, result);

        cv::imshow("test", result);
        cv::waitKey(2);

        cur_angle += cur_step;
        if (cur_angle >= max_blade_angle)
            cur_step = -cur_step;
        if (cur_angle <= min_blade_angle)
            cur_step = -cur_step;
        video_writer.write(blade_image);
    }
}