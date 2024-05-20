/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:31
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-17 10:47:10
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
//
// Created by touch on 22-11-29.
//

#include "common/EcalImageSender.h"

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
 * @brief 打开Ecal图像发送器并初始化
 *
 * 该函数用于初始化Ecal图像发送器，如果eCAL尚未初始化，则会先进行初始化。
 * 并启用循环回路，创建一个指向指定主题的protobuf发布者的智能指针。
 *
 * @param topic 将要发送图像的主题名称
 */
void EcalImageSender::open(const std::string& topic) {
    // 检查eCAL是否已经初始化，如果未初始化，则进行初始化
    if (!eCAL::IsInitialized()) {
        eCAL::Initialize();
    }
    // 启用循环回路，确保发送的消息可以被本地接收
    eCAL::Util::EnableLoopback(true);
    // 创建一个protobuf发布者，用于发送OpencvImage格式的数据
    m_pub_image =
        std::make_shared<eCAL::protobuf::CPublisher<xcmg_proto::OpencvImage>>(
            topic);
}
