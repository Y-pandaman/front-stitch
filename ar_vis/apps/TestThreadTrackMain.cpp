#include "ThreadInteractiveWidget.h"
#include <Eigen/Eigen>
#include <QApplication>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    ThreadInteractiveWidget thread_interactive_widget;
    printf("start thread_interactive_widget main work thread\n");
    thread_interactive_widget.startMainWorkerThread(argc, argv);
    while (!thread_interactive_widget.checkHaveNewImage())
        ;
    cv::Mat ar_image = thread_interactive_widget.getNewestImage();
    cv::Mat result;
    int frame_count = 0;
    while (true) {
        static cv::Mat img2 = cv::imread(
            "/home/ps/workspace/stitcher/data/1226_calib_ground_/1226_calib_ground_0/noblur_camera_video_1/1-0.png");
        printf("main() checkHaveNewImage..\n");
        bool flag = thread_interactive_widget.checkHaveNewImage();
        if (flag) {
            printf("main() frame_count = %d getNewestImage()..\n", frame_count);
            ar_image = thread_interactive_widget.getNewestImage();
        }

        cv::addWeighted(ar_image, 0.5, img2, 0.5, 0, result);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        if (flag)
            cv::imwrite("./output/track/" + std::to_string(frame_count) +
                            ".png",
                        result);
        frame_count++;
    }
}