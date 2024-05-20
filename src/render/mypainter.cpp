#include "render/mypainter.h"

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

/**
 * @brief 绘制赛道
 *
 * 该函数用于使用OpenGL绘制给定的赛道。它能够根据提供的投影、视图和模型矩阵，以及赛道的颜色和其它属性来渲染赛道。
 * 如果需要，还可以绘制车轮轨迹。
 *
 * @param track 要绘制的赛道对象，包含赛道的几何和属性信息。
 * @param projection 投影矩阵，用于将3D空间中的物体投影到2D屏幕上。
 * @param view 视图矩阵，表示摄像机的位置和朝向。
 * @param model 模型矩阵，表示物体在世界空间中的位置、旋转和缩放。
 * @param track_color 赛道的颜色。
 * @param min_show_v 显示赛道的最小速度值。
 * @param draw_wheel_track 是否绘制车轮轨迹。
 * @param tex_offset_y 贴图在Y轴上的偏移量。
 */
void MyPainter::drawTrack(Track& track, const QMatrix4x4& projection,
                          const QMatrix4x4& view, const QMatrix4x4& model,
                          const QVector3D& track_color, float min_show_v,
                          bool draw_wheel_track, float tex_offset_y) {
    // 确保OpenGL上下文为当前上下文，并绑定赛道的VAO以准备绘制
    widget_->makeCurrent();
    track_vao_->bind();

    // 绑定并配置赛道的VBO，包含赛道的顶点数据
    track_vbo_->bind();
    track_vbo_->allocate(track.getPointList(), track.getBufferSize());
    track_shader->bind();

    // 根据是否绘制车轮轨迹，选择合适的纹理进行绑定
    if (draw_wheel_track) {
        wheel_track_texture->setWrapMode(QOpenGLTexture::WrapMode::Repeat);
        wheel_track_texture->bind();
    } else {
        blade_track_texture->setWrapMode(QOpenGLTexture::WrapMode::Repeat);
        blade_track_texture->bind();
    }

    // 启用并配置顶点属性，包括位置和纹理坐标
    track_shader->shader_program_.enableAttributeArray("aPos");
    track_shader->shader_program_.setAttributeBuffer(
        "aPos", GL_FLOAT, 0, TRACK_POS_COORD_LEN,
        TRACK_POINT_LEN * sizeof(float));
    track_shader->shader_program_.enableAttributeArray("aTexCoord");
    track_shader->shader_program_.setAttributeBuffer(
        "aTexCoord", GL_FLOAT, TRACK_POS_COORD_LEN * sizeof(float),
        TRACK_TEXTURE_COORD_LEN, TRACK_POINT_LEN * sizeof(float));

    // 设置着色器中使用的uniform变量，包括矩阵和颜色
    track_shader->setUniformValue("projection", projection);
    track_shader->setUniformValue("view", view);
    track_shader->setUniformValue("model", model);

    track_shader->setUniformValue("tex_offset_y", tex_offset_y);

    track_shader->setUniformValue("track_color", track_color);
    track_shader->setUniformValue("min_show_v", min_show_v);
    track_shader->setUniformValue("max_show_v", 26);

    // 获取并使用OpenGL 4.3的核心函数来绘制轨迹
    QOpenGLFunctions_4_3_Core* gl_function_ptr =
        QOpenGLContext::currentContext()
            ->versionFunctions<QOpenGLFunctions_4_3_Core>();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // 执行绘制命令
    gl_function_ptr->glDrawArrays(GL_TRIANGLES, 0, track.getPointNum());
    glDisable(GL_BLEND);   // 关闭混合

    // 解除绑定并释放资源
    track_shader->unbind();
    track_vbo_->release();
    track_vao_->release();
}

/**
 * @brief 绘制轨迹回放
 *
 * 该函数用于使用给定的轨道数据、投影矩阵、视图矩阵和模型矩阵，以及轨道颜色和最小显示速度，
 * 在关联的OpenGL窗口上绘制轨道的回放效果。
 *
 * @param track 轨道对象，包含要绘制的轨道数据
 * @param projection 投影矩阵，用于将3D坐标转换为2D屏幕坐标
 * @param view 视图矩阵，表示观察者的视图方向和位置
 * @param model 模型矩阵，代表物体在世界坐标系中的位置、旋转和缩放
 * @param track_color 轨道颜色，用于着色绘制的轨道
 * @param min_show_v 最小显示速度，用于控制轨道点的可见性
 */
void MyPainter::drawBackTrack(Track& track, const QMatrix4x4& projection,
                              const QMatrix4x4& view, const QMatrix4x4& model,
                              const QVector3D& track_color, float min_show_v) {
    // 切换到对应的OpenGL上下文
    widget_->makeCurrent();
    // 绑定着色器程序
    back_track_shader->bind();
    // 绑定顶点数组对象
    track_vao_->bind();

    // 绑定顶点缓冲对象并更新其内容为轨道点数据
    track_vbo_->bind();
    track_vbo_->allocate(track.getPointList(), track.getBufferSize());

    // 启用着色器中的属性数组，并设置属性缓冲
    back_track_shader->shader_program_.enableAttributeArray("aPos");
    back_track_shader->shader_program_.setAttributeBuffer(
        "aPos", GL_FLOAT, 0, TRACK_POS_COORD_LEN,
        TRACK_POINT_LEN * sizeof(float));
    back_track_shader->shader_program_.enableAttributeArray("aTexCoord");
    back_track_shader->shader_program_.setAttributeBuffer(
        "aTexCoord", GL_FLOAT, TRACK_POS_COORD_LEN * sizeof(float),
        TRACK_TEXTURE_COORD_LEN, TRACK_POINT_LEN * sizeof(float));
    // 设置着色器中的统一变量
    back_track_shader->setUniformValue("projection", projection);
    back_track_shader->setUniformValue("view", view);
    back_track_shader->setUniformValue("model", model);
    back_track_shader->setUniformValue("track_color", track_color);
    back_track_shader->setUniformValue("min_show_v", min_show_v);

    // 配置混合功能以实现透明效果
    QOpenGLFunctions_4_3_Core* gl_function_ptr =
        QOpenGLContext::currentContext()
            ->versionFunctions<QOpenGLFunctions_4_3_Core>();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    // 绘制轨道
    gl_function_ptr->glDrawArrays(GL_TRIANGLES, 0, track.getPointNum());
    glDisable(GL_BLEND);   // 关闭混合功能

    // 解除着色器和缓冲对象的绑定
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

/**
 * @brief 绘制文本
 *
 * 此函数用于使用给定的坐标、投影、视图和模型矩阵以及文本颜色绘制文本。
 * 它首先绑定并分配文本的VBO（顶点缓冲对象），然后启用着色器程序并设置必要的uniform值，
 * 包括投影、视图、模型矩阵以及文本颜色。接着，它绘制通过VBO提供的文本顶点。
 *
 * @param text_point_list
 * 包含文本顶点坐标的浮点数向量。每个文本顶点由多个坐标组成。
 * @param projection 投影矩阵，用于将3D坐标投影到2D屏幕空间。
 * @param view 视图矩阵，表示观察者的视图方向和位置。
 * @param model 模型矩阵，表示物体在世界空间中的位置、尺寸和旋转。
 * @param text_color 文本颜色，以三维向量表示（红、绿、蓝）。
 */
void MyPainter::drawText(const std::vector<float>& text_point_list,
                         const QMatrix4x4& projection, const QMatrix4x4& view,
                         const QMatrix4x4& model, const QVector3D& text_color) {
    // 切换到对应的QOpenGLWidget上下文
    widget_->makeCurrent();

    // 绑定并分配文本的顶点缓冲对象
    text_vbo_->bind();
    text_vbo_->allocate(text_point_list.data(),
                        text_point_list.size() * sizeof(float));

    // 确保着色器程序已准备就绪
    assert(text_shader != nullptr);

    // 绑定着色器程序
    text_shader->bind();
    // 绑定并设置纹理参数
    QOpenGLTexture* texture = track_label_texture;
    texture->setWrapMode(QOpenGLTexture::WrapMode::Repeat);
    texture->bind();

    // 启用着色器中的属性数组，并设置属性缓冲
    text_shader->shader_program_.enableAttributeArray("aPos");
    text_shader->shader_program_.setAttributeBuffer(
        "aPos", GL_FLOAT, 0, TRACK_POS_COORD_LEN,
        TRACK_POINT_LEN * sizeof(float));
    text_shader->shader_program_.enableAttributeArray("aTexCoord");
    text_shader->shader_program_.setAttributeBuffer(
        "aTexCoord", GL_FLOAT, TRACK_POS_COORD_LEN * sizeof(float),
        TRACK_TEXTURE_COORD_LEN, TRACK_POINT_LEN * sizeof(float));

    // 设置着色器中的uniform变量
    text_shader->setUniformValue("projection", projection);
    text_shader->setUniformValue("view", view);
    text_shader->setUniformValue("model", model);
    text_shader->setUniformValue("text_color", text_color);

    // 获取并使用4.3版本的核心OpenGL函数，设置混合函数并绘制文本
    QOpenGLFunctions_4_3_Core* gl_function_ptr =
        QOpenGLContext::currentContext()
            ->versionFunctions<QOpenGLFunctions_4_3_Core>();
    gl_function_ptr->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    gl_function_ptr->glDrawArrays(GL_TRIANGLES, 0,
                                  text_point_list.size() / TRACK_POINT_LEN);

    // 解除着色器程序的绑定并释放顶点缓冲对象
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

/**
 * 绘制模型
 *
 * 该函数负责遍历模型中的所有网格，并调用drawMesh函数绘制每一个网格。
 * 使用了OpenGL进行绘制，需要传入模型数据、投影矩阵、视图矩阵、模型矩阵以及模型颜色。
 *
 * @param model 模型数据，包含多个网格。
 * @param projection 投影矩阵，用于将3D世界中的物体投影到2D屏幕上。
 * @param view 视图矩阵，代表了相机的位置和方向。
 * @param model_matrix 模型矩阵，用于放置和旋转模型。
 * @param model_color 模型颜色，为整个模型指定一个基本颜色。
 */
void MyPainter::drawModel(const Model& model, const QMatrix4x4& projection,
                          const QMatrix4x4& view,
                          const QMatrix4x4& model_matrix,
                          const QVector3D& model_color) {
    // 遍历模型中的所有网格，并对每个网格进行绘制
    for (const auto& mesh : model.meshes) {
        drawMesh(mesh, projection, view, model_matrix, model_color);
    }
}

/**
 * 绘制网格对象。
 *
 * @param mesh 要绘制的网格，包含顶点、索引和顶点数组对象等信息。
 * @param projection 投影矩阵，用于将3D世界中的物体投影到2D屏幕上。
 * @param view 视图矩阵，表示摄像机的位置和方向。
 * @param model 模型矩阵，表示物体在世界坐标系中的位置、旋转和缩放。
 * @param model_color 模型的颜色。
 */
void MyPainter::drawMesh(const Mesh& mesh, const QMatrix4x4& projection,
                         const QMatrix4x4& view, const QMatrix4x4& model,
                         const QVector3D& model_color) {
    // 切换到绘制使用的OpenGL上下文
    widget_->makeCurrent();

    // 获取网格的顶点缓冲对象、索引缓冲对象和顶点数组对象
    QOpenGLBuffer *qvbo = mesh.get_qvbo(), *qebo = mesh.get_qebo();
    QOpenGLVertexArrayObject* qvao = mesh.get_qvao();

    // 绑定着色器程序和顶点数组对象
    mesh_shader->bind();
    qvao->bind();

    // 设置着色器中使用的矩阵参数（投影、视图、模型）
    mesh_shader->setUniformValue("projection", projection);
    mesh_shader->setUniformValue("view", view);
    mesh_shader->setUniformValue("model", model);

    // 设置视图位置、光源位置和颜色等着色器参数
    QVector3D tp;
    tp = QVector3D(view.column(3));
    mesh_shader->setUniformValue("view_pos", tp);
    tp = QVector3D(0, 10, 10);   // 设置光源位置
    mesh_shader->setUniformValue("light_pos", tp);
    tp = QVector3D(1, 1, 1);   // 设置光源颜色
    mesh_shader->setUniformValue("light_color", tp);
    mesh_shader->setUniformValue("object_color", model_color);

    // 获取并启用OpenGL 4.3的核心功能指针，用于绘制
    QOpenGLFunctions_4_3_Core* gl_function_ptr =
        QOpenGLContext::currentContext()
            ->versionFunctions<QOpenGLFunctions_4_3_Core>();
    glEnable(GL_DEPTH_TEST);   // 启用深度测试

    // 执行绘制命令
    gl_function_ptr->glDrawElements(GL_TRIANGLES, mesh.indices.size(),
                                    GL_UNSIGNED_INT, 0);
    glDisable(GL_DEPTH_TEST);   // 禁用深度测试
    mesh_shader->unbind();      // 解除着色器绑定
    qvao->release();            // 释放顶点数组对象
}
