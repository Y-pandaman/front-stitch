#pragma once

#include <QGLShaderProgram>
#include <QKeyEvent>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLBuffer>
#include <QOpenGLFramebufferObject>
#include <QOpenGLVertexArrayObject>
#include <QVector3D>
#include <QVector4D>
#include <QVector>
#include <QtOpenGL>
#include <cmath>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vector_types.h>

static int glCheckError_(const char* file, int line) {
    int errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR) {
        std::string error;
        switch (errorCode) {
        case GL_INVALID_ENUM:
            error = "INVALID_ENUM";
            break;
        case GL_INVALID_VALUE:
            error = "INVALID_VALUE";
            break;
        case GL_INVALID_OPERATION:
            error = "INVALID_OPERATION";
            break;
        case GL_STACK_OVERFLOW:
            error = "STACK_OVERFLOW";
            break;
        case GL_STACK_UNDERFLOW:
            error = "STACK_UNDERFLOW";
            break;
        case GL_OUT_OF_MEMORY:
            error = "OUT_OF_MEMORY";
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            error = "INVALID_FRAMEBUFFER_OPERATION";
            break;
        }
        std::cout << error << " | " << file << " (" << line << ")" << std::endl;
        std::system("pause");
    }
    return errorCode;
}
#define glCheckError() glCheckError_(__FILE__, __LINE__)

class FrameBuffer {
public:
    FrameBuffer() : fbo_(NULL), fbo_ms_(NULL), width_(0), height_(0) { }
    ~FrameBuffer() {
        if (!fbo_) {
            delete fbo_;
            fbo_ = NULL;
        }
        if (!fbo_ms_) {
            delete fbo_ms_;
            fbo_ms_ = NULL;
        }
        if (cuda_res_ != nullptr) {
            cudaGraphicsUnregisterResource(cuda_res_);
            delete m_dev_ptr;
        }
    }

    /**
     * 函数名称：bind
     * 功能描述：绑定当前的多采样FBO（如果存在）或者普通FBO（如果多采样FBO不存在）。
     * 参数列表：无
     * 返回值：无
     */
    void bind() {
        // 尝试绑定多采样FBO，如果不存在，则尝试绑定普通FBO
        if (fbo_ms_) {
            fbo_ms_->bind();   // 绑定多采样FBO
        } else {
            if (fbo_) {
                fbo_->bind();   // 绑定普通FBO
            }
        }
    }

    /**
     * @brief 解绑当前的多采样FBO（如果存在）或者普通FBO。
     *
     * 该函数首先尝试释放多采样FBO（fbo_ms_），如果它存在的话。如果多采样FBO不存在，则尝试释放普通FBO（fbo_）。
     * 这个函数通常在不再需要使用FBO时调用，以确保资源得到正确释放，防止内存泄漏。
     */
    void unbind() {
        // 首先尝试释放多采样FBO
        if (fbo_ms_) {
            fbo_ms_->release();
        } else {
            // 如果多采样FBO不存在，则尝试释放普通FBO
            if (fbo_) {
                fbo_->release();
            }
        }
    }

    void resize(GLenum internal_format, int width, int height,
                bool bind_cuda = false, bool multisample = false) {
        if (!fbo_ || width != width_ || height != height_) {
            if (!fbo_) {
                delete fbo_;
            }

            width_           = width;
            height_          = height;
            size_in_byte     = width * height * 4 * sizeof(float);
            internal_format_ = internal_format;
            multisample_     = multisample;
            fbo_             = new QGLFramebufferObject(
                width, height,
                QGLFramebufferObject::Attachment::CombinedDepthStencil,
                GL_TEXTURE_2D, internal_format);
            if (multisample) {
                QGLFramebufferObjectFormat format;
                format.setSamples(16);
                format.setInternalTextureFormat(internal_format);
                format.setAttachment(
                    QGLFramebufferObject::Attachment::CombinedDepthStencil);
                format.setTextureTarget(GL_TEXTURE_2D_MULTISAMPLE);
                fbo_ms_ = new QGLFramebufferObject(width, height, format);
            }
            glCheckError();
        }
        if (bind_cuda) {
            cudaGraphicsGLRegisterImage(&cuda_res_, fbo_->texture(),
                                        GL_TEXTURE_2D,
                                        cudaGraphicsRegisterFlagsNone);
            cudaMalloc((void**)&m_dev_ptr, size_in_byte);
        }
    }

    bool updateCudaMem();
    float* getCudaPtr();

    /**
     * 下载纹理图像数据到内存。
     *
     * @param data 指向接收纹理数据的内存缓冲区的指针。
     * @param format 指定下载的纹理数据的格式。
     * @param type 指定下载的纹理数据的类型。
     */
    void downloadTexture(uchar* data, GLenum format, GLenum type) {
        // 如果存在有效的FBO对象，则绑定该纹理并下载其图像数据
        if (fbo_) {
            glBindTexture(GL_TEXTURE_2D, fbo_->texture());   // 绑定纹理
            glGetTexImage(GL_TEXTURE_2D, 0, format, type,
                          data);               // 下载纹理图像数据
            glBindTexture(GL_TEXTURE_2D, 0);   // 解除纹理绑定
            glCheckError();                    // 检查OpenGL错误
        }
    }

    /**
     * 将多采样FBO（MSAA）的内容.blit（复制）到普通FBO中。
     * 这个操作主要用于将多采样抗锯齿效果应用到一个普通的帧缓冲对象上。
     *
     * @param f 指向QOpenGLExtraFunctions的指针，用于调用OpenGL的额外功能。
     */
    void bliteFboMSToFbo(QOpenGLExtraFunctions* f) {
        // 如果多采样FBO存在，则进行.blit操作
        if (fbo_ms_) {
            f->glBindFramebuffer(GL_READ_FRAMEBUFFER,
                                 fbo_ms_->handle());   // 绑定读取帧缓冲
            f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER,
                                 fbo_->handle());   // 绑定绘制帧缓冲
            // 执行.blit操作，从多采样FBO拷贝到普通FBO
            f->glBlitFramebuffer(0, 0, width_, height_, 0, 0, width_, height_,
                                 GL_COLOR_BUFFER_BIT, GL_NEAREST);
            // 解除对帧缓冲的绑定
            f->glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glCheckError();   // 检查OpenGL错误
        }
    }

public:
    QGLFramebufferObject* fbo_;
    QGLFramebufferObject* fbo_ms_;
    int width_, height_, size_in_byte;
    GLenum internal_format_;
    bool multisample_;

    float* m_dev_ptr                 = nullptr;
    cudaGraphicsResource_t cuda_res_ = nullptr;
    cudaArray_t cuda_array_          = nullptr;
};

class ShaderToTexture {
public:
    ShaderToTexture(const char* vertex_shader_file,
                    const char* fragment_shader_file, bool is_compile_from_file,
                    FrameBuffer* fb)
        : fb_(fb) {
        QGLShader* vertex_shader   = new QGLShader(QGLShader::Vertex);
        QGLShader* fragment_shader = new QGLShader(QGLShader::Fragment);
        if (is_compile_from_file) {
            vertex_shader->compileSourceFile(vertex_shader_file);
            fragment_shader->compileSourceFile(fragment_shader_file);
        } else {
            vertex_shader->compileSourceCode(vertex_shader_file);
            fragment_shader->compileSourceCode(fragment_shader_file);
        }
        shader_program_.addShader(vertex_shader);
        shader_program_.addShader(fragment_shader);
        shader_program_.link();
        shader_program_.removeAllShaders();
    }

    ShaderToTexture() : fb_(NULL) { }

    ~ShaderToTexture() { }

    void bindAndClear() {
        if (fb_->width_ == 0 || fb_->height_ == 0) {
            std::cout << "need to resize the fb first" << std::endl;
            std::system("pause");
            std::exit(0);
        }

        if (fb_->fbo_ms_) {
            fb_->fbo_ms_->bind();
        } else {
            if (fb_->fbo_) {
                fb_->fbo_->bind();
            }
        }
        shader_program_.bind();
        glPushAttrib(GL_VIEWPORT_BIT);
        glViewport(0, 0, fb_->width_, fb_->height_);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glCheckError();
    }
    void bind() {
        if (fb_->width_ == 0 || fb_->height_ == 0) {
            std::cout << "need to resize the fb first" << std::endl;
            std::system("pause");
            std::exit(0);
        }

        if (fb_->fbo_ms_) {
            fb_->fbo_ms_->bind();
        } else {
            if (fb_->fbo_) {
                fb_->fbo_->bind();
            }
        }
        shader_program_.bind();
        glPushAttrib(GL_VIEWPORT_BIT);
        glViewport(0, 0, fb_->width_, fb_->height_);
        glCheckError();
    }

    void unbind() {
        glPopAttrib();
        shader_program_.release();
        if (fb_->fbo_ms_) {
            fb_->fbo_ms_->release();
        } else {
            if (fb_->fbo_) {
                fb_->fbo_->release();
            }
        }
        glCheckError();
    }

    void renderToViewport(QOpenGLExtraFunctions* f, float4 view_port,
                          GLuint framebuffer) {
        // Blite multisample VBO to normal VBO
        f->glBindFramebuffer(GL_READ_FRAMEBUFFER, fb_->fbo_->handle());
        f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer);
        f->glBlitFramebuffer(0, 0, fb_->width_, fb_->height_, view_port.x,
                             view_port.y, view_port.z, view_port.w,
                             GL_COLOR_BUFFER_BIT, GL_NEAREST);
        f->glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        f->glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
        glCheckError();
    }

    /**
     * @brief 设置着色器程序中的统一变量的值。
     *
     * 该函数模板用于为着色器程序中的统一变量设置值。它通过名称找到对应的统一变量，并赋予指定的值。
     *
     * @param name 统一变量的名称。这是一个const
     * char*类型，指向统一变量名称的字符串。
     * @param value
     * 要赋予统一变量的值。这个模板函数支持多种类型，使得可以为不同类型的统一变量设置值。
     */
    template <typename T> void setUniformValue(const char* name, T value) {
        shader_program_.setUniformValue(
            name, value);   // 调用shader_program_的成员函数设置统一变量的值
    }

public:
    QGLShaderProgram shader_program_;
    FrameBuffer* fb_;
};
