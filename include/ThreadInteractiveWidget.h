/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-20 15:12:42
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#ifndef CYLINDER_STITCHER_THREADINTERACTIVEWIDGET_H
#define CYLINDER_STITCHER_THREADINTERACTIVEWIDGET_H

#include "common/EcalBladeHeightSender.h"
#include "can.pb.h"
#include "common/EcalImageSender.h"
#include "interactiveimagewidget.h"
#include "steering_angle.pb.h"
#include "util/innoreal_timer.hpp"
#include <atomic>
#include <condition_variable>
#include <ecal/ecal.h>
#include <opencv2/opencv.hpp>

#define THREAD_INTERACTIVE_WIDGET_IMAGE_BUF_SIZE 2

//#define OUTPUT_BLADE_IMAGE
//#define OUTPUT_BLADE_VIDEO

//#define OUTPUT_BACK_TRACK_IMAGE

class ThreadInteractiveWidget {
private:
    std::shared_ptr<eCAL::protobuf::CSubscriber<xcmg_proto::Grader_CAN>>
        m_sub_steering_angle;

    std::atomic<float> current_angle      = 0;
    std::atomic_bool current_angle_update = true, have_new_image = false;

    void ecalSteeringAngleCallBack(const char* _topic_name,
                                   const xcmg_proto::Grader_CAN& msg,
                                   long long _time, long long _clock,
                                   long long _id);

    std::atomic_int l_cg = 1053, l_bd = 1061, l_ef = 1096, blade_theta = 0,
                    l_IJ = 492, l_LM = 3450;

    bool startSteeringAngleEcalReceiver(const std::string& topic_name,
                                        int queue_size_);

    void mainWorker(int argc, char** argv);

    std::thread* main_worker_thread = nullptr;

    EcalImageSender blade_image_sender, back_track_image_sender;
    EcalBladeHeightSender blade_height_sender;

public:
    ~ThreadInteractiveWidget() {
        if (main_worker_thread != nullptr)
            main_worker_thread->join();
    }

    bool startMainWorkerThread(int argc, char** argv);

    /**
     * 检查是否有新的图像数据可用
     *
     * 本函数用于查询系统或某个数据源中是否有新的图像数据可供使用。
     *
     * @param 无
     * @return bool 返回一个布尔值，若存在新的图像数据则为true，否则为false。
     */
    bool checkHaveNewImage() {
        return have_new_image;   // 返回是否有新的图像数据标志
    }

    InteractiveImageWidget* widget_ptr = nullptr;

    float* getTrackImageCudaPtr();
};

#endif   // CYLINDER_STITCHER_THREADINTERACTIVEWIDGET_H
