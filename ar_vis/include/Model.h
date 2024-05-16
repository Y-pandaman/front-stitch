//
// Created by touch on 22-11-21.
//

#ifndef CYLINDER_STITCHER_MODEL_H
#define CYLINDER_STITCHER_MODEL_H

#include "shaders.h"
#include <Eigen/Eigen>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <assert.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <filesystem>
#include <iostream>
#include <vector>

struct Vertex {
    Eigen::Vector3f position, normal;
    Eigen::Vector2f tex_coords;
};
struct Texture {
    uint id;
    std::string type;
};

class Mesh {
private:
    QOpenGLBuffer *qvbo, *qebo;
    QOpenGLVertexArrayObject* qvao;

public:
    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    std::vector<Texture> textures;

    Mesh(std::vector<Vertex> _vertices, std::vector<uint> _indices,
         std::vector<Texture> _textures);

    void setupMesh(QOpenGLBuffer* _qvbo, QOpenGLVertexArrayObject* _qvao,
                   QOpenGLBuffer* _qveo, ShaderToTexture* shader);

    QOpenGLBuffer* get_qvbo() const {
        return qvbo;
    }

    QOpenGLBuffer* get_qebo() const {
        return qebo;
    }

    QOpenGLVertexArrayObject* get_qvao() const {
        return qvao;
    }
};

class Model {
private:
    void initTransform(Eigen::Matrix3f R, Eigen::Vector3f pos);

    Mesh processMesh(aiMesh* mesh, const aiScene* scene,
                     const Eigen::Matrix3f& R   = Eigen::Matrix3f::Identity(),
                     const Eigen::Vector3f& pos = Eigen::Vector3f(0, 0, 0));

    void processNode(aiNode* node, const aiScene* scene,
                     const Eigen::Matrix3f& R   = Eigen::Matrix3f::Identity(),
                     const Eigen::Vector3f& pos = Eigen::Vector3f(0, 0, 0));

    bool inited = false;

    Eigen::Matrix4f model_matrix = Eigen::Matrix4f::Identity();

public:
    std::vector<Mesh> meshes;

    Vertex getVertex(int idx);

    Model(const std::filesystem::path& model_file_path);

    Model();

    void loadModel(const std::filesystem::path& _model_path,
                   const Eigen::Matrix3f& R   = Eigen::Matrix3f::Identity(),
                   const Eigen::Vector3f& pos = Eigen::Vector3f(0, 0, 0));

    /**
     * 获取当前对象中网格的数量。
     *
     * 本函数不接受任何参数。
     *
     * @return 返回一个无符号整数（uint），表示网格的数量。
     */
    uint getMeshesNum() const {
        return meshes.size();   // 返回存储网格的容器的大小
    }

    /**
     * 设置模型矩阵
     * @param m
     * 模型矩阵，是一个4x4的齐次变换矩阵，用于将模型坐标系中的点转换到世界坐标系中。
     */
    void setModelMatrix(const Eigen::Matrix4f& m) {
        model_matrix = m;   // 更新模型矩阵为传入的矩阵
    }

    Eigen::Matrix4f getModelMatrix() {
        return model_matrix;
    }

    QMatrix4x4 getModelQMatrix();
};

#endif   // CYLINDER_STITCHER_MODEL_H
