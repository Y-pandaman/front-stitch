/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-20 16:33:26
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "common/GstReceiver.h"
#include "core/Config.h"
#include "core/FrontStitcherMain.h"
#include <opencv2/opencv.hpp>
#include <thread>

extern Config config;

int main(int argc, char** argv) {
    std::thread tp(&frontStitcherMain, argc, argv);
    tp.detach();

    while (1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    return 0;
}