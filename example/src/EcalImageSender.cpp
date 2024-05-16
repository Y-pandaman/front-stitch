/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-09 18:52:35
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "EcalImageSender.h"
#include "include/EcalImageSender.h"

EcalImageSender::EcalImageSender(int argc, char** argv) { }

EcalImageSender::~EcalImageSender() { }

void EcalImageSender::pubImage(cv::Mat image) {
    xcmg_proto::OpencvImage message_opencv_image;

    message_opencv_image.set_rows(image.rows);
    message_opencv_image.set_cols(image.cols);
    message_opencv_image.set_elt_type(image.type());
    size_t data_size = image.rows * image.cols * image.elemSize();
    message_opencv_image.set_mat_data(image.data, data_size);

    m_pub_image->Send(message_opencv_image);
}

void EcalImageSender::open(const std::string& topic) {
    if (!eCAL::IsInitialized()) {
        eCAL::Initialize();
    }
    eCAL::Util::EnableLoopback(true);
    m_pub_image =
        std::make_shared<eCAL::protobuf::CPublisher<xcmg_proto::OpencvImage>>(
            topic);

    m_pub_image->SetLayerMode(eCAL::TLayer::tlayer_all,
                              eCAL::TLayer::smode_off);
    m_pub_image->SetLayerMode(eCAL::TLayer::tlayer_tcp,
                              eCAL::TLayer::smode_on);   // 使用TCP
    m_pub_image->SetLayerMode(eCAL::TLayer::tlayer_inproc,
                              eCAL::TLayer::smode_auto);
    m_pub_image->SetLayerMode(eCAL::TLayer::tlayer_shm,
                              eCAL::TLayer::smode_auto);
}
