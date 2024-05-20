#ifndef CYLINDER_STITCHER_ECALBLADEHEIGHTSENDER_H
#define CYLINDER_STITCHER_ECALBLADEHEIGHTSENDER_H

#include "blade_height.pb.h"
#include <ecal/ecal.h>
#include <ecal/msg/protobuf/publisher.h>

class EcalBladeHeightSender {
public:
    EcalBladeHeightSender(int argc = 0, char** argv = nullptr);
    ~EcalBladeHeightSender();

    void open(const std::string& topic);
    void pubMessage(float left_height, float right_height);

private:
    std::shared_ptr<eCAL::protobuf::CPublisher<xcmg_proto::blade_height>> m_pub;
};

#endif   // CYLINDER_STITCHER_ECALBLADEHEIGHTSENDER_H
