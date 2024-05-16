#include "Model.h"
#include <utility>

/**
 * 加载模型函数
 *
 * 通过给定的文件路径，旋转矩阵和位置向量加载模型。
 * 使用Assimp库进行模型的加载和处理，将模型的几何形状和材质信息存储在内部结构中。
 *
 * @param _model_path 模型文件的文件系统路径。
 * @param R 旋转矩阵，用于模型加载后的空间转换。
 * @param pos 位置向量，同样用于模型加载后的空间转换。
 */
void Model::loadModel(const std::filesystem::path& _model_path,
                      const Eigen::Matrix3f& R, const Eigen::Vector3f& pos) {
    // 检查模型是否已经初始化，如果已经初始化，则直接返回
    if (inited) {
        printf("model is inited\n");
        return;
    }
    inited = true;
    Assimp::Importer importer;

    // 使用Assimp库读取模型文件
    const aiScene* scene = importer.ReadFile(
        _model_path, aiProcess_Triangulate | aiProcess_FlipUVs);

    // 检查模型是否成功加载
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
        !scene->mRootNode) {
        std::cout << "ERROR::ASSIMP::" << importer.GetErrorString()
                  << std::endl;
        return;
    }
    // 递归处理模型的节点，进行空间变换和网格合并
    processNode(scene->mRootNode, scene, R, pos);

    // 特定模型路径下的调试代码，用于输出某些顶点的信息
    if (_model_path.string().compare("/home/touch/xugong/3505_ply/2.1.ply") ==
        0) {
        printf("in loadModel\n");
        int left_idx = 402893, right_idx = 431068;
        std::cout << left_idx << "\n"
                  << meshes[0].vertices[left_idx].position << std::endl;
        std::cout << right_idx << "\n"
                  << meshes[0].vertices[right_idx].position << std::endl;
        //        exit(0);
    }
}

/**
 * 处理一个AI节点，包括该节点的所有网格和其子节点。
 *
 * @param node 指向当前正在处理的aiNode的指针。
 * @param scene 包含整个场景信息的aiScene结构体指针。
 * @param R 当前节点的旋转矩阵，用于将AI空间中的坐标转换为应用中的坐标。
 * @param pos 当前节点的位置向量，同上用于坐标转换。
 */
void Model::processNode(aiNode* node, const aiScene* scene,
                        const Eigen::Matrix3f& R, const Eigen::Vector3f& pos) {
    // 遍历当前节点的所有网格，将它们处理并添加到meshes列表中
    for (int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        meshes.push_back(processMesh(mesh, scene, R, pos));
    }

    // 遍历当前节点的所有子节点，并对它们递归调用processNode函数
    for (int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene, R, pos);
    }
}

/**
 * 处理并加载一个三维模型的网格数据。
 *
 * @param mesh 指向aiMesh的指针，代表一个三维模型的网格。
 * @param scene 指向aiScene的指针，代表整个三维模型的场景信息。
 * @param R 一个3x3的旋转矩阵，用于将网格数据从原始坐标系旋转到目标坐标系。
 * @param pos 一个3维向量，表示目标坐标系的原点位置。
 * @return 返回一个Mesh对象，包含处理后的网格数据，包括顶点、索引和纹理。
 */
Mesh Model::processMesh(aiMesh* mesh, const aiScene* scene,
                        const Eigen::Matrix3f& R, const Eigen::Vector3f& pos) {
    std::vector<Vertex> vertices;   // 存储顶点信息的容器
    std::vector<uint> indices;   // 存储顶点索引的容器，用于绘制
    std::vector<Texture> textures;   // 存储纹理信息的容器

    // 遍历网格中的所有顶点，收集顶点信息
    for (uint i = 0; i < mesh->mNumVertices; i++) {
        Vertex vertex;
        // 处理顶点位置，先缩放后旋转，最后平移
        vertex.position =
            Eigen::Vector3f(mesh->mVertices[i].x, mesh->mVertices[i].y,
                            mesh->mVertices[i].z) /
            1000.0;   // 缩放因子，取决于原始数据的单位
        vertex.position =
            R.transpose() * (vertex.position - pos);   // 旋转和平移
        // 如果有法线信息，则处理法线
        if (mesh->HasNormals()) {
            vertex.normal = -Eigen::Vector3f(
                mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
            vertex.normal = R.transpose() * vertex.normal;   // 旋转法线
        }
        // 如果有纹理坐标，则处理纹理坐标
        if (mesh->mTextureCoords[0]) {
            vertex.tex_coords = Eigen::Vector2f(mesh->mTextureCoords[0][i].x,
                                                mesh->mTextureCoords[0][i].y);
        } else {
            // 如果没有纹理坐标，则默认为(0,0)
            vertex.tex_coords = Eigen::Vector2f(0, 0);
        }
        vertices.push_back(vertex);
    }

    // 遍历网格中的所有面，收集索引信息
    for (uint i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        for (uint j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }

    // 如果网格有材质信息，则进行处理（当前代码段留空，待实现）
    if (mesh->mMaterialIndex >= 0) {
        // TODO materials
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
    }

    return {vertices, indices, textures};   // 返回处理后的网格数据
}

Model::Model(const std::filesystem::path& model_file_path) {
    loadModel(model_file_path);
}

void Model::initTransform(Eigen::Matrix3f R, Eigen::Vector3f pos) { }

Model::Model() {
    inited = false;
}

/**
 * 获取模型的QMatrix4x4矩阵。
 * 该函数不接受参数。
 *
 * @return 返回一个QMatrix4x4类型的矩阵，该矩阵是模型矩阵的副本。
 */
QMatrix4x4 Model::getModelQMatrix() {
    QMatrix4x4 model_qmatrix;
    // 将模型矩阵的数据复制到QMatrix4x4中
    memcpy(model_qmatrix.data(), model_matrix.data(), 16 * sizeof(float));
    return model_qmatrix;
}

/**
 * 从模型中获取指定索引的顶点
 * @param idx 顶点的索引
 * @return 返回指定索引的顶点
 */
Vertex Model::getVertex(int idx) {
    // 确保模型中只有一个网格
    assert(meshes.size() == 1);
    // 确保指定索引不越界
    assert(meshes[0].vertices.size() > idx);
    return meshes[0].vertices[idx];
}

Mesh::Mesh(std::vector<Vertex> _vertices, std::vector<uint> _indices,
           std::vector<Texture> _textures)
    : vertices(std::move(_vertices)), indices(std::move(_indices)),
      textures(std::move(_textures)) { }

/**
 * @brief
 * 设置Mesh的渲染数据，包括顶点缓冲对象（VBO）、顶点数组对象（VAO）、索引缓冲对象（EBO）和着色器。
 * @param _qvbo 指向QOpenGLBuffer的指针，用于存储顶点数据。
 * @param _qvao
 * 指向QOpenGLVertexArrayObject的指针，用于简化渲染过程中的状态设置。
 * @param _qveo 指向QOpenGLBuffer的指针，用于存储索引数据。
 * @param shader 指向ShaderToTexture的指针，包含用于渲染的着色器程序。
 */
void Mesh::setupMesh(QOpenGLBuffer* _qvbo, QOpenGLVertexArrayObject* _qvao,
                     QOpenGLBuffer* _qveo, ShaderToTexture* shader) {
    // 初始化VAO、VBO和EBO
    qvao = _qvao;
    qvao->create();
    qvbo = _qvbo;
    qvbo->create();
    qebo = _qveo;
    qebo->create();

    // 绑定VAO，准备设置顶点属性
    qvao->bind();
    qvbo->bind();
    // 设置顶点缓冲的使用模式为StaticDraw，并分配内存
    qvbo->setUsagePattern(QOpenGLBuffer::StaticDraw);
    qvbo->allocate(vertices.data(), vertices.size() * sizeof(Vertex));

    // 绑定着色器程序，并设置顶点属性
    shader->shader_program_.bind();
    // 启用并设置位置属性
    shader->shader_program_.enableAttributeArray("aPos");
    shader->shader_program_.setAttributeBuffer("aPos", GL_FLOAT, 0, 3,
                                               8 * sizeof(float));
    // 启用并设置法向量属性
    shader->shader_program_.enableAttributeArray("aNormal");
    shader->shader_program_.setAttributeBuffer(
        "aNormal", GL_FLOAT, 3 * sizeof(float), 3, 8 * sizeof(float));
    // 启用并设置纹理坐标属性
    shader->shader_program_.enableAttributeArray("aTexCoords");
    shader->shader_program_.setAttributeBuffer(
        "aTexCoords", GL_FLOAT, 6 * sizeof(float), 2, 8 * sizeof(float));
    // 绑定EBO并设置使用模式及内存分配
    qebo->bind();
    qebo->setUsagePattern(QOpenGLBuffer::StaticDraw);
    qebo->allocate(indices.data(), indices.size() * sizeof(uint));

    // 释放绑定，防止资源泄露
    qvao->release();
    qvbo->release();
    qebo->release();
    shader->shader_program_.release();
}
