/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:31
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-20 15:12:51
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */

#include "common/EcalBladeHeightSender.h"

/**
 * @brief 初始化EcalBladeHeightSender，建立发布者并绑定指定的主题。
 *
 * @param topic 指定的主题名称，用于发布blade_height消息。
 */
void EcalBladeHeightSender::open(const std::string& topic) {
    // 检查eCAL是否已初始化，如未初始化则进行初始化
    if (!eCAL::IsInitialized()) {
        eCAL::Initialize();
    }
    // 启用循环回环，以支持消息的本地发布和订阅
    eCAL::Util::EnableLoopback(true);
    // 创建并绑定一个基于主题的protobuf发布者
    m_pub =
        std::make_shared<eCAL::protobuf::CPublisher<xcmg_proto::blade_height>>(
            topic);
}

/**
 * @brief 发送blade_height消息至指定的主题。
 *
 * @param left_height 左侧刀片的高度。
 * @param right_height 右侧刀片的高度。
 */
void EcalBladeHeightSender::pubMessage(float left_height, float right_height) {
    // 创建一个新的blade_height消息，并设置左右刀片的高度
    xcmg_proto::blade_height message;
    message.set_left_height(left_height);
    message.set_right_height(right_height);
    // 使用发布者发送设置好的消息
    m_pub->Send(message);
}

EcalBladeHeightSender::EcalBladeHeightSender(int argc, char** argv) { }

EcalBladeHeightSender::~EcalBladeHeightSender() { }
