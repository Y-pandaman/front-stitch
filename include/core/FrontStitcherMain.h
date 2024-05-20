/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-20 17:30:35
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#ifndef CYLINDER_STITCHER_FRONTSTITCHERMAIN_H
#define CYLINDER_STITCHER_FRONTSTITCHERMAIN_H

#include "common/EcalImageSender.h"
#include "common/GstReceiver.h"
#include "common/cylinder_stitcher.cuh"
#include "stage/ThreadInteractiveWidget.h"
#include "stage/interactiveimagewidget.h"
#include "util/innoreal_timer.hpp"
#include <Eigen/Eigen>
#include <QApplication>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <stack>

// #define USE_INPUT_RESIZE_FACTOR

//#define USE_720P_TO_1080P

int frontStitcherMain(int argc, char** argv);

#endif   // CYLINDER_STITCHER_FRONTSTITCHERMAIN_H
