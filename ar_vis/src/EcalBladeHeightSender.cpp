//
// Created by touch on 23-10-31.
//

#include "EcalBladeHeightSender.h"

void EcalBladeHeightSender::open(const std::string& topic) {
    if (!eCAL::IsInitialized()) {
        eCAL::Initialize();
    }
    eCAL::Util::EnableLoopback(true);
    m_pub =
        std::make_shared<eCAL::protobuf::CPublisher<xcmg_proto::blade_height>>(
            topic);
}

void EcalBladeHeightSender::pubMessage(float left_height, float right_height) {
    xcmg_proto::blade_height message;
    message.set_left_height(left_height);
    message.set_right_height(right_height);
    m_pub->Send(message);
}

EcalBladeHeightSender::EcalBladeHeightSender(int argc, char** argv) { }

EcalBladeHeightSender::~EcalBladeHeightSender() { }
