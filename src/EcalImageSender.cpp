/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-17 10:45:27
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "EcalImageSender.h"
#include "include/EcalImageSender.h"

EcalImageSender::EcalImageSender(int argc, char** argv) { }

EcalImageSender::~EcalImageSender() { }

/**
 * @brief 发布OpenCV图像
 *
 * 这个函数将给定的OpenCV图像打包成自定义消息格式，然后通过一个图像发布者发送出去。
 *
 * @param image 要发布的OpenCV图像。图像数据将被直接从这个参数中读取并发送。
 */
void EcalImageSender::pubImage(cv::Mat image) {
    // 创建自定义消息对象，并设置图像的基本信息（行数、列数、元素类型）
    xcmg_proto::OpencvImage message_opencv_image;

    message_opencv_image.set_rows(image.rows);
    message_opencv_image.set_cols(image.cols);
    message_opencv_image.set_elt_type(image.type());

    // 计算图像数据的大小，并将图像数据复制到消息对象中
    size_t data_size = image.rows * image.cols * image.elemSize();
    message_opencv_image.set_mat_data(image.data, data_size);

    // 发送图像消息
    m_pub_image->Send(message_opencv_image);
}

/**
 * @brief 打开Ecal图片发送器并配置发布者
 *
 * 该函数主要用于初始化Ecal图片发送器，检查并初始化eCAL库，配置发布者，
 * 包括启用循环回路，设置网络层模式等。
 *
 * @param topic 要发布的主题名称
 */
void EcalImageSender::open(const std::string& topic) {
    // 检查eCAL是否已初始化，如未初始化则进行初始化
    if (!eCAL::IsInitialized()) {
        eCAL::Initialize();
    }
    // 启用循环回路，确保本地发布的消息可以被本地订阅者接收
    eCAL::Util::EnableLoopback(true);

    // 创建并配置OpencvImage类型的protobuf发布者
    m_pub_image =
        std::make_shared<eCAL::protobuf::CPublisher<xcmg_proto::OpencvImage>>(
            topic);

    // 设置发布者的网络层模式
    // 全部禁用，只启用TCP和自动选择inproc及shm
    m_pub_image->SetLayerMode(eCAL::TLayer::tlayer_all,
                              eCAL::TLayer::smode_off);
    m_pub_image->SetLayerMode(eCAL::TLayer::tlayer_tcp,
                              eCAL::TLayer::smode_on);   // 启用TCP层
    m_pub_image->SetLayerMode(eCAL::TLayer::tlayer_inproc,
                              eCAL::TLayer::smode_auto);
    m_pub_image->SetLayerMode(eCAL::TLayer::tlayer_shm,
                              eCAL::TLayer::smode_auto);
}
