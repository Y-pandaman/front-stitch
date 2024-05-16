#include "mypainter.h"

/**
 * MyPainter的构造函数
 * 用于初始化绘画对象，包括帧缓存对象、驱动辅助对象、以及各种OpenGL顶点和纹理对象。
 *
 * @param fb_track 指向跟踪画面帧缓存的指针。
 * @param fb_blade 指向刀片画面帧缓存的指针。
 * @param fb_back_track 指向背面跟踪画面帧缓存的指针。
 * @param widget 指向OpenGL小部件的指针，用于OpenGL渲染。
 */
MyPainter::MyPainter(FrameBuffer* fb_track, FrameBuffer* fb_blade,
                     FrameBuffer* fb_back_track, QOpenGLWidget* widget)
    : fb_track_(fb_track), fb_blade_(fb_blade), fb_back_track_(fb_back_track),
      widget_(widget) {
    // 获取应用目录路径
    QString applicationdDirPath;
    applicationdDirPath = QCoreApplication::applicationDirPath();
    // 初始化驱动辅助对象
    drive_assistant =
        new DriveAssistant(3.506, 6.405, 5.0, 60.0, 0.1, 1000);   // 3505

    // 创建OpenGL缓冲区对象，用于存储顶点数据
    track_vbo_ = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
    rect_vbo_  = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);
    text_vbo_  = new QOpenGLBuffer(QOpenGLBuffer::Type::VertexBuffer);

    // 创建OpenGL顶点数组对象，用于组织和管理顶点数据
    track_vao_ = new QOpenGLVertexArrayObject(widget_);
    // 确保顶点数组对象正确创建
    if (track_vao_->isCreated()) {
        track_vao_->destroy();
    }
    track_vao_->create();

    rect_vao_ = new QOpenGLVertexArrayObject(widget_);
    // 确保顶点数组对象正确创建
    if (rect_vao_->isCreated()) {
        rect_vao_->destroy();
    }
    rect_vao_->create();

    // 加载并创建OpenGL纹理对象，用于绘制时的纹理映射
    QImage temp;
    temp.load(config.wheel_track_img_path);   // 轨迹图像
    wheel_track_texture = new QOpenGLTexture(temp);

    temp.load(config.blade_track_img_path);   // 刀片轨迹图像
    blade_track_texture = new QOpenGLTexture(temp);

    temp.load(config.track_label_texture_path);   // 轨迹标签纹理
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

/**
 * 初始化绘制轨迹的相关资源。
 * 该函数负责初始化用于绘制轨迹的着色器、顶点缓冲对象（VBO）和顶点数组对象（VAO）。
 * 它首先检查并创建所需的着色器程序，然后准备顶点数据的存储结构。
 */
void MyPainter::initializeDrawTrack() {
    widget_->makeCurrent();   // 确保当前是所需的QOpenGLWidget上下文
    // 如果轨迹着色器未初始化，则创建它
    if (track_shader == NULL) {
        // 定义顶点着色器代码
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

        // 定义片段着色器代码
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

        // 定义后轨迹片段着色器代码（用于背景轨迹）
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
        // 创建着色器对象并关联它们
        track_shader = new ShaderToTexture(vshader_char_arr, fshader_char_arr,
                                           false, fb_track_);
        back_track_shader =
            new ShaderToTexture(vshader_char_arr, back_track_fshader_char_arr,
                                false, fb_back_track_);
        glCheckError();   // 检查OpenGL调用错误
    }

    // 销毁并重新创建顶点缓冲对象，以确保最新的数据被使用
    if (track_vbo_->isCreated()) {
        track_vbo_->destroy();
    }
    track_vbo_->create();
    track_vbo_->setUsagePattern(
        QOpenGLBuffer::DynamicDraw);   // 设置VBO使用模式为动态绘制
    // 绑定并释放顶点数组对象，以配置其属性
    track_vao_->bind();
    track_vao_->release();

    glCheckError();   // 再次检查OpenGL调用错误
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

/**
 * 初始化绘制文本的相关设置。
 * 该函数配置文本的源管理器，设置着色器和纹理缓冲对象（VBO）用于文本渲染。
 *
 * 无参数
 * 无返回值
 */
void MyPainter::initializeDrawText() {
    // 初始化文本源管理器，使用配置中指定的字体像素高度
    text_source_manager.initialize(config.font_pixel_height);
    // 断言驾驶助手对象不为空
    assert(drive_assistant != nullptr);
    // 将文本源管理器设置给驾驶助手的左轨道
    drive_assistant->left_track.setTextSourceManager(&text_source_manager);
    // 使当前窗口为绘图目标
    widget_->makeCurrent();

    // 定义并初始化顶点着色器的字符数组
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

    // 定义并初始化片段着色器的字符数组
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

    // 如果文本着色器对象为空，则创建一个新的着色器对象
    if (text_shader == nullptr) {
        text_shader = new ShaderToTexture(vshader_char_arr, fshader_char_arr,
                                          false, fb_track_);
        // 检查OpenGL错误
        glCheckError();
    }

    // 如果文本VBO已经创建，则先销毁，准备重新创建
    if (text_vbo_->isCreated()) {
        text_vbo_->destroy();
    }
    // 创建并设置纹理缓冲对象的使用模式为动态绘制
    text_vbo_->create();
    text_vbo_->setUsagePattern(QOpenGLBuffer::DynamicDraw);

    // 绑定纹理缓冲对象
    text_vbo_->bind();

    // 检查OpenGL错误
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

/**
 * 初始化绘制模型
 * 该函数遍历模型中的所有网格，并对每个网格调用initializeDrawMesh函数进行初始化设置。
 *
 * @param model 引用传递的模型对象，包含待绘制的几何体信息。
 */
void MyPainter::initializeDrawModel(Model& model) {
    // 获取模型中网格的数量
    uint meshes_num = model.getMeshesNum();
    // 遍历所有网格，对每个网格进行初始化设置
    for (int i = 0; i < meshes_num; i++) {
        initializeDrawMesh(model.meshes[i]);
    }
}

/**
 * 初始化绘制网格的着色器和其他资源。
 *
 * @param mesh 要绘制的网格，其渲染所需的数据将在此函数中被设置。
 */
void MyPainter::initializeDrawMesh(Mesh& mesh) {
    widget_->makeCurrent();   // 确保当前是绘图目标widget的上下文

    // 如果着色器对象尚未创建，则创建并编译着色器
    if (mesh_shader == nullptr) {
        // 定义顶点着色器代码
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

        // 定义片段着色器代码
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

        // 创建新的着色器程序并关联到当前的绘制网格
        mesh_shader = new ShaderToTexture(vshader_char_arr, fshader_char_arr,
                                          false, fb_blade_);
        glCheckError();   // 检查着色器编译过程是否出错
    }

    // 设置网格的渲染资源，包括顶点缓冲对象、顶点数组对象和索引缓冲对象
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
