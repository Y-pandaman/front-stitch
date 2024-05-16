/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-09 18:51:48
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */

#ifndef CYLINDER_STITCHER_GSTRECEIVER_H
#define CYLINDER_STITCHER_GSTRECEIVER_H

#include "EcalImageSender.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

class GstReceiver {
public:
    ~GstReceiver();
    bool initialize(const std::string& url_, int queue_size_ = 2);
    bool startReceiveWorker();
    uchar* getImageData();
    cv::Mat getImageMat();

    void openEcalImageTopic(const std::string& ecal_topic_name);
    void setFlagEcalSend(bool flag) {
        flag_ecal_send = flag;
    }

private:
    int frame_count = 0;
    void receiveWorker();
    std::condition_variable cond_image_queue_not_empty;
    std::atomic_bool stop_flag = false;
    cv::VideoCapture video_capture;
    int queue_size;
    int video_width, video_height;
    std::string video_url;
    cv::Mat* image_queue               = nullptr;
    std::mutex* mutex_on_image_queue   = nullptr;
    std::thread *thread_receive_worker = nullptr,
                *thread_write_worker   = nullptr;
    std::atomic_ullong p_write = 0, p_read = 0;

    std::atomic_bool flag_ecal_send;
    EcalImageSender ecal_image_sender;

    enum Status {
        UNINITIALIZED,
        INITIALIZED,
        RUN,
        STOP
    } status = UNINITIALIZED;
};

#endif   // CYLINDER_STITCHER_GSTRECEIVER_H
