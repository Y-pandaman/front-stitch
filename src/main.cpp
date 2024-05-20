/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-20 14:55:16
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#include "FrontStitcherMain.h"
#include "common/GstReceiver.h"
#include "include/Config.h"
#include <opencv2/opencv.hpp>
#include <thread>

extern Config config;

int main(int argc, char** argv) {
    std::thread tp(&frontStitcherMain, argc, argv);
    tp.join();

    return 0;
}