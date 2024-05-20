/*
 * @Author: 姚潘涛
 * @Date: 2024-05-08 10:14:32
 * @LastEditors: 姚潘涛
 * @LastEditTime: 2024-05-16 16:38:20
 * @Description:
 *
 * Copyright (c) 2024 by pandaman, All Rights Reserved.
 */
#ifndef MYPAINTER_H
#define MYPAINTER_H

#include "Model.h"
#include "TagPosesSet.h"
#include "TextSourceManager.h"
#include "driveassistant.h"
#include "shaders.h"
#include "util/math_utils.h"
#include <QCoreApplication>
#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QOpenGLFunctions>
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLShader>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QtWidgets/QOpenGLWidget>

#define MYSHADER_VERTEX_ATTRIBUTE 0
#define MYSHADER_TEXCOORD_ATTRIBUTE 1

#define TRACKSHADER_VERTEX_ATTRIBUTE 0

class MyPainter {
public:
    MyPainter(FrameBuffer* fb_track, FrameBuffer* fb_blade,
              FrameBuffer* fb_back_track, QOpenGLWidget* widget);

    ~MyPainter();

    // 0 for left 1 for right
    void drawTrack(Track& track, const QMatrix4x4& projection,
                   const QMatrix4x4& view, const QMatrix4x4& model,
                   const QVector3D& track_color, float min_show_v,
                   bool draw_wheel_track, float tex_offset_y);
    void drawBackTrack(Track& track, const QMatrix4x4& projection,
                       const QMatrix4x4& view, const QMatrix4x4& model,
                       const QVector3D& track_color, float min_show_v);

    void drawText(const std::vector<float>& text_point_list,
                  const QMatrix4x4& projection, const QMatrix4x4& view,
                  const QMatrix4x4& model, const QVector3D& text_color);

    void drawModel(const Model& model, const QMatrix4x4& projection,
                   const QMatrix4x4& view, const QMatrix4x4& model_matrix,
                   const QVector3D& model_color);

    void drawMesh(const Mesh& mesh, const QMatrix4x4& projection,
                  const QMatrix4x4& view, const QMatrix4x4& model,
                  const QVector3D& model_color);

    void initializeDrawTrack();

    void initializeDrawText();

    void initializeDrawModel(Model& model);

    void initializeDrawRects();

    /**
     * 获取DriveAssistant对象的指针。
     *
     * 该函数不接受任何参数。
     *
     * @return 返回一个指向DriveAssistant对象的指针。
     */
    DriveAssistant* getDriveAssistant() {
        return drive_assistant;
    }

private:
    void initializeDrawMesh(Mesh& mesh);

    QOpenGLBuffer *track_vbo_, *rect_vbo_, *text_vbo_;
    QOpenGLVertexArrayObject *track_vao_, *rect_vao_;
    DriveAssistant* drive_assistant = nullptr;
    ShaderToTexture *track_shader = nullptr, *back_track_shader = nullptr,
                    *cross_shader = nullptr, *rect_shader = nullptr,
                    *text_shader = nullptr, *mesh_shader = nullptr;

    FrameBuffer *fb_track_, *fb_blade_, *fb_back_track_;
    QOpenGLWidget* widget_;

    TextSourceManager text_source_manager;

    QOpenGLTexture *wheel_track_texture, *blade_track_texture;
    QOpenGLTexture* track_label_texture;
};

#endif   // MYPAINTER_H
