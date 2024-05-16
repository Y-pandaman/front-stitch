#include "mypainter.h"

MyPainter::MyPainter(FrameBuffer* fb_track, FrameBuffer* fb_blade,
                     FrameBuffer* fb_back_track, QOpenGLWidget* widget)
    : fb_track_(fb_track), fb_blade_(fb_blade), fb_back_track_(fb_back_track),
      widget_(widget) {
    //在mainwindow.h中声明变量，最好是全局变量啦，
    QString applicationdDirPath;
    applicationdDirPath = QCoreApplication::applicationDirPath();
    drive_assistant =
        new DriveAssistant(3.506, 6.405, 5.0, 60.0, 0.1, 1000);   // 3505

    track_vbo_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
    rect_vbo_  = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
    text_vbo_  = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);

    track_vao_ = new QOpenGLVertexArrayObject(widget_);
    if (track_vao_->isCreated()) {
        track_vao_->destroy();
    }
    track_vao_->create();

    rect_vao_ = new QOpenGLVertexArrayObject(widget_);
    if (rect_vao_->isCreated()) {
        rect_vao_->destroy();
    }
    rect_vao_->create();

    QImage temp;
    temp.load(config.wheel_track_img_path);
    wheel_track_texture = new QOpenGLTexture(temp);

    temp.load(config.blade_track_img_path);
    blade_track_texture = new QOpenGLTexture(temp);

    temp.load(config.track_label_texture_path);
    track_label_texture = new QOpenGLTexture(temp);
}

MyPainter::~MyPainter() { }

void MyPainter::drawTrack(Track& track, const QMatrix4x4& projection,
                          const QMatrix4x4& view, const QMatrix4x4& model,
                          const QVector3D& track_color, float min_show_v,
                          bool draw_wheel_track, float tex_offset_y) {
    widget_->makeCurrent();
    track_vao_->bind();

    track_vbo_->bind();
    track_vbo_->allocate(track.getPointList(), track.getBufferSize());
    track_shader->bind();

    if (draw_wheel_track) {
        wheel_track_texture->setWrapMode(QOpenGLTexture::WrapMode::Repeat);
        wheel_track_texture->bind();
    } else {
        blade_track_texture->setWrapMode(QOpenGLTexture::WrapMode::Repeat);
        blade_track_texture->bind();
    }

    track_shader->shader_program_.enableAttributeArray("aPos");
    track_shader->shader_program_.setAttributeBuffer(
        "aPos", GL_FLOAT, 0, TRACK_POS_COORD_LEN,
        TRACK_POINT_LEN * sizeof(float));
    track_shader->shader_program_.enableAttributeArray("aTexCoord");
    track_shader->shader_program_.setAttributeBuffer(
        "aTexCoord", GL_FLOAT, TRACK_POS_COORD_LEN * sizeof(float),
        TRACK_TEXTURE_COORD_LEN, TRACK_POINT_LEN * sizeof(float));

    track_shader->setUniformValue("projection", projection);
    track_shader->setUniformValue("view", view);
    track_shader->setUniformValue("model", model);

    track_shader->setUniformValue("tex_offset_y", tex_offset_y);

    track_shader->setUniformValue("track_color", track_color);
    track_shader->setUniformValue("min_show_v", min_show_v);
    track_shader->setUniformValue("max_show_v", 26);
    QOpenGLFunctions_4_3_Core* gl_function_ptr =
        QOpenGLContext::currentContext()
            ->versionFunctions<QOpenGLFunctions_4_3_Core>();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    gl_function_ptr->glDrawArrays(GL_TRIANGLES, 0, track.getPointNum());
    glDisable(GL_BLEND);
    track_shader->unbind();
    track_vbo_->release();
    track_vao_->release();
}

void MyPainter::drawBackTrack(Track& track, const QMatrix4x4& projection,
                              const QMatrix4x4& view, const QMatrix4x4& model,
                              const QVector3D& track_color, float min_show_v) {
    widget_->makeCurrent();
    back_track_shader->bind();
    track_vao_->bind();

    track_vbo_->bind();
    track_vbo_->allocate(track.getPointList(), track.getBufferSize());

    back_track_shader->shader_program_.enableAttributeArray("aPos");
    back_track_shader->shader_program_.setAttributeBuffer(
        "aPos", GL_FLOAT, 0, TRACK_POS_COORD_LEN,
        TRACK_POINT_LEN * sizeof(float));
    back_track_shader->shader_program_.enableAttributeArray("aTexCoord");
    back_track_shader->shader_program_.setAttributeBuffer(
        "aTexCoord", GL_FLOAT, TRACK_POS_COORD_LEN * sizeof(float),
        TRACK_TEXTURE_COORD_LEN, TRACK_POINT_LEN * sizeof(float));
    back_track_shader->setUniformValue("projection", projection);
    back_track_shader->setUniformValue("view", view);
    back_track_shader->setUniformValue("model", model);

    back_track_shader->setUniformValue("track_color", track_color);
    back_track_shader->setUniformValue("min_show_v", min_show_v);
    QOpenGLFunctions_4_3_Core* gl_function_ptr =
        QOpenGLContext::currentContext()
            ->versionFunctions<QOpenGLFunctions_4_3_Core>();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    gl_function_ptr->glDrawArrays(GL_TRIANGLES, 0, track.getPointNum());
    glDisable(GL_BLEND);
    back_track_shader->unbind();
    track_vbo_->release();
    track_vao_->release();
}

void MyPainter::initializeDrawTrack() {
    widget_->makeCurrent();
    if (track_shader == NULL) {
        char vshader_char_arr[] =
            "#version 430 core\n"
            "in vec3 aPos;\n"
            "in vec2 aTexCoord;\n"
            "out vec2 TexCoord;\n"
            "uniform vec2 image_size;\n"
            "uniform mat4x4 matrix_plane2image;\n"
            "uniform mat4x4 transform_matrix;\n"
            "uniform mat4x4 projection, view, model;\n"
            "void main(void){\n"
            "    gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
            "    TexCoord = aTexCoord;\n"
            "    TexCoord.y = - (aTexCoord.y);\n"
            "}\n";

        char fshader_char_arr[] =
            "#version 430 core\n"
            "out vec4 fragColor;\n"
            "in vec2 TexCoord;\n"
            "uniform vec3 track_color;\n"
            "layout(binding=0) uniform sampler2D track_texture;\n"
            "uniform float min_show_v;\n"
            "uniform float max_show_v;"
            "uniform float tex_offset_y;\n"
            "void main(void){\n"
            "    float fade_len = 0.5;\n"
            "    vec2 dy_tex_coord = TexCoord;\n"
            "    float alpha_factor = 2.0;"
            "    if (-TexCoord.y < min_show_v + fade_len) {alpha_factor = (-TexCoord.y - min_show_v) / fade_len;}\n"
            "    if (-TexCoord.y > 28 - fade_len) {alpha_factor = (28 + TexCoord.y) / fade_len;}\n"
            "    alpha_factor = min(1.0, max(0, alpha_factor));\n"
            "    dy_tex_coord.y = dy_tex_coord.y + tex_offset_y;\n"
            "    vec4 sampled = texture(track_texture, dy_tex_coord);\n"
            "    sampled.a *= alpha_factor;\n"
            "    fragColor = sampled;\n"
            "}\n";

        char back_track_fshader_char_arr[] =
            "#version 430 core\n"
            "out vec4 fragColor;\n"
            "in vec2 TexCoord;\n"
            "uniform vec3 track_color;\n"
            "layout(binding=0) uniform sampler2D track_texture;\n"
            "uniform float min_show_v;\n"
            "uniform float max_show_v;"
            "uniform float tex_offset_y;\n"
            "void main(void){\n"
            "    fragColor = vec4(track_color, 1.0);\n"
            "    if(-TexCoord.y < min_show_v) fragColor = vec4(0.0, 0.0, 0.0, 0.0);\n"
            "}\n";
        track_shader = new ShaderToTexture(vshader_char_arr, fshader_char_arr,
                                           false, fb_track_);
        back_track_shader =
            new ShaderToTexture(vshader_char_arr, back_track_fshader_char_arr,
                                false, fb_back_track_);
        glCheckError();
    }

    if (track_vbo_->isCreated()) {
        track_vbo_->destroy();
    }
    track_vbo_->create();
    track_vbo_->setUsagePattern(QOpenGLBuffer::DynamicDraw);
    track_vao_->bind();
    track_vao_->release();

    glCheckError();
}
void MyPainter::initializeDrawRects() {
    widget_->makeCurrent();

    char vshader_char_arr[] = "#version 430 core\n"
                              "in vec2 points;\n"
                              "void main(void){\n"
                              "    gl_Position = vec4(points, 1.0f, 1.0f);\n"
                              "}\n";

    char fshader_char_arr[] = "#version 430 core\n"
                              "out vec4 fragColor;\n"
                              "void main(void){\n"
                              "    fragColor = vec4(1.0, 0.0, 0.0, 0.0);\n"
                              "}\n";

    if (rect_shader == NULL) {
        rect_shader = new ShaderToTexture(vshader_char_arr, fshader_char_arr,
                                          false, fb_track_);
        glCheckError();
    }

    if (rect_vbo_->isCreated()) {
        rect_vbo_->destroy();
    }
    rect_vbo_->create();
    rect_vbo_->setUsagePattern(QOpenGLBuffer::DynamicDraw);

    rect_vao_->bind();

    rect_vbo_->bind();
    rect_vbo_->allocate(100 * 4 * 2 * sizeof(GLfloat));
    rect_shader->shader_program_.bind();
    rect_shader->shader_program_.enableAttributeArray("points");
    rect_shader->shader_program_.setAttributeBuffer("points", GL_FLOAT, 0, 2);
    rect_shader->shader_program_.release();
    rect_vbo_->release();

    rect_vao_->release();
    glCheckError();
}

void MyPainter::initializeDrawText() {
    text_source_manager.initialize(config.font_pixel_height);
    assert(drive_assistant != nullptr);
    drive_assistant->left_track.setTextSourceManager(&text_source_manager);
    widget_->makeCurrent();

    char vshader_char_arr[] =
        "#version 430 core\n"
        "in vec3 aPos;\n"
        "in vec2 aTexCoord;\n"
        "out vec2 TexCoord;\n"
        "uniform mat4x4 projection, view, model;\n"
        "void main(void){\n"
        "    gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
        "    TexCoord = aTexCoord;\n"
        "}\n";

    char fshader_char_arr[] =
        "#version 430 core\n"
        "layout(binding=0) uniform sampler2D text_texture;\n"
        "out vec4 fragColor;\n"
        "in vec2 TexCoord;\n"
        "uniform vec3 text_color;\n"
        "void main(void){\n"
        "    vec4 sampled = texture(text_texture, TexCoord);\n"
        "    fragColor = sampled;\n"
        "}\n";

    if (text_shader == nullptr) {
        text_shader = new ShaderToTexture(vshader_char_arr, fshader_char_arr,
                                          false, fb_track_);
        glCheckError();
    }

    if (text_vbo_->isCreated()) {
        text_vbo_->destroy();
    }
    text_vbo_->create();
    text_vbo_->setUsagePattern(QOpenGLBuffer::DynamicDraw);

    text_vbo_->bind();

    glCheckError();
}

void MyPainter::drawText(const std::vector<float>& text_point_list,
                         const QMatrix4x4& projection, const QMatrix4x4& view,
                         const QMatrix4x4& model, const QVector3D& text_color) {
    widget_->makeCurrent();

    text_vbo_->bind();
    text_vbo_->allocate(text_point_list.data(),
                        text_point_list.size() * sizeof(float));
    assert(text_shader != nullptr);

    text_shader->bind();
    QOpenGLTexture* texture = track_label_texture;
    texture->setWrapMode(QOpenGLTexture::WrapMode::Repeat);
    texture->bind();

    text_shader->shader_program_.enableAttributeArray("aPos");
    text_shader->shader_program_.setAttributeBuffer(
        "aPos", GL_FLOAT, 0, TRACK_POS_COORD_LEN,
        TRACK_POINT_LEN * sizeof(float));
    text_shader->shader_program_.enableAttributeArray("aTexCoord");
    text_shader->shader_program_.setAttributeBuffer(
        "aTexCoord", GL_FLOAT, TRACK_POS_COORD_LEN * sizeof(float),
        TRACK_TEXTURE_COORD_LEN, TRACK_POINT_LEN * sizeof(float));

    text_shader->setUniformValue("projection", projection);
    text_shader->setUniformValue("view", view);
    text_shader->setUniformValue("model", model);
    text_shader->setUniformValue("text_color", text_color);

    QOpenGLFunctions_4_3_Core* gl_function_ptr =
        QOpenGLContext::currentContext()
            ->versionFunctions<QOpenGLFunctions_4_3_Core>();
    gl_function_ptr->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    gl_function_ptr->glDrawArrays(GL_TRIANGLES, 0,
                                  text_point_list.size() / TRACK_POINT_LEN);

    text_shader->unbind();
    text_vbo_->release();
}

void MyPainter::initializeDrawModel(Model& model) {
    uint meshes_num = model.getMeshesNum();
    for (int i = 0; i < meshes_num; i++) {
        initializeDrawMesh(model.meshes[i]);
    }
}

void MyPainter::initializeDrawMesh(Mesh& mesh) {
    widget_->makeCurrent();

    if (mesh_shader == nullptr) {
        char vshader_char_arr[] =
            "#version 430 core\n"
            "in vec3 aPos;\n"
            "in vec3 aNormal;\n"
            "in vec2 aTexCoords;\n"
            "out vec3 Normal;\n"
            "out vec3 frag_pos;\n"
            "uniform mat4x4 projection, model, view;\n"
            "void main(void){\n"
            "    Normal = mat3(transpose(inverse(model))) * aNormal;\n"
            "    frag_pos = vec3(model * vec4(aPos, 1));\n"
            "    gl_Position = projection * view * model * vec4(aPos, 1.0);\n"
            "}\n";

        char fshader_char_arr[] =
            "#version 430 core\n"
            "uniform vec3 view_pos;\n"
            "uniform vec3 light_pos;\n"
            "uniform vec3 light_color;\n"
            "uniform vec3 object_color;\n"
            "in vec3 Normal;\n"
            "in vec3 frag_pos;\n"
            "out vec4 fragColor;\n"
            "void main(void){\n"

            "    float ambient_strength = 0.3;\n"
            "    vec3 ambient = ambient_strength * light_color;\n"
            ""
            "    vec3 N = (normalize(Normal));\n"
            "    vec3 light_dir = normalize(frag_pos - light_pos);\n"
            "    float diff = max(dot(N, light_dir), 0.0);\n"
            "    vec3 diffuse = diff * light_color;\n"

            "    float specular_strength = 0.5;\n"
            "    vec3 view_dir = normalize(frag_pos - view_pos);\n"
            "    vec3 reflect_dir = reflect(-light_dir, N);\n"
            "    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);\n"
            "    vec3 specular = specular_strength * spec * light_color;\n"
            ""
            "    vec3 result = (ambient + diffuse + specular) * object_color;\n"
            "    fragColor = vec4(result, 1.0);\n"
            "}\n";
        mesh_shader = new ShaderToTexture(vshader_char_arr, fshader_char_arr,
                                          false, fb_blade_);
        glCheckError();
    }

    mesh.setupMesh(new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer),
                   new QOpenGLVertexArrayObject(this->widget_),
                   new QOpenGLBuffer(QOpenGLBuffer::Type::IndexBuffer),
                   mesh_shader);
}

void MyPainter::drawModel(const Model& model, const QMatrix4x4& projection,
                          const QMatrix4x4& view,
                          const QMatrix4x4& model_matrix,
                          const QVector3D& model_color) {
    for (const auto& mesh : model.meshes) {
        drawMesh(mesh, projection, view, model_matrix, model_color);
    }
}

void MyPainter::drawMesh(const Mesh& mesh, const QMatrix4x4& projection,
                         const QMatrix4x4& view, const QMatrix4x4& model,
                         const QVector3D& model_color) {
    widget_->makeCurrent();

    QOpenGLBuffer *qvbo = mesh.get_qvbo(), *qebo = mesh.get_qebo();
    QOpenGLVertexArrayObject* qvao = mesh.get_qvao();

    mesh_shader->bind();
    qvao->bind();

    mesh_shader->setUniformValue("projection", projection);
    mesh_shader->setUniformValue("view", view);
    mesh_shader->setUniformValue("model", model);

    QVector3D tp;
    tp = QVector3D(view.column(3));
    mesh_shader->setUniformValue("view_pos", tp);
    tp = QVector3D(0, 10, 10);
    mesh_shader->setUniformValue("light_pos", tp);
    tp = QVector3D(1, 1, 1);
    mesh_shader->setUniformValue("light_color", tp);
    mesh_shader->setUniformValue("object_color", model_color);
    QOpenGLFunctions_4_3_Core* gl_function_ptr =
        QOpenGLContext::currentContext()
            ->versionFunctions<QOpenGLFunctions_4_3_Core>();
    glEnable(GL_DEPTH_TEST);

    gl_function_ptr->glDrawElements(GL_TRIANGLES, mesh.indices.size(),
                                    GL_UNSIGNED_INT, 0);
    glDisable(GL_DEPTH_TEST);
    mesh_shader->unbind();
    qvao->release();
}
