#include "BladeModelTransformer.h"

std::vector<QVector3D> BladeModelTransformer::trilateration(const QVector3D& P1,
                                                            const QVector3D& P2,
                                                            const QVector3D& P3,
                                                            float r1, float r2,
                                                            float r3) {
    QVector3D p1(0, 0, 0);
    QVector3D p2 = P2 - P1;
    QVector3D p3 = P3 - P1;
    QVector3D v1 = p2 - p1;
    QVector3D v2 = p3 - p1;

    QVector3D Xn  = v1.normalized();
    QVector3D tmp = QVector3D::crossProduct(v1, v2);
    QVector3D Zn  = tmp.normalized();

    QVector3D Yn =
        QVector3D::crossProduct(Xn, Zn);   // TODO check (Xn, Zn) or (Zn, Xn)

    float i = QVector3D::dotProduct(Xn, v2);
    float d = QVector3D::dotProduct(Xn, v1);
    float j = QVector3D::dotProduct(Yn, v2);

    float X = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
    float Y = (((r1 * r1) - (r3 * r3) + (i * i) + (j * j)) / (2 * j)) -
              ((i / j) * (X));
    float Z1 = std::sqrt(std::max(0.0f, r1 * r1 - X * X - Y * Y));
    float Z2 = -Z1;

    std::vector<QVector3D> result(2);
    result[0] = P1 + X * Xn + Y * Yn + Z1 * Zn;
    result[1] = P1 + X * Xn + Y * Yn + Z2 * Zn;

    return result;
}

/**
 * 执行三边测量定位，根据三个已知点和对应半径计算未知点的位置。
 *
 * @param P1 第一个已知点的坐标（向量形式）。
 * @param P2 第二个已知点的坐标（向量形式）。
 * @param P3 第三个已知点的坐标（向量形式）。
 * @param r1 第一个已知点的半径。
 * @param r2 第二个已知点的半径。
 * @param r3 第三个已知点的半径。
 * @return 计算得到的未知点的位置（向量形式）。
 */
Eigen::Vector3f BladeModelTransformer::trilateration(const Eigen::Vector3f& P1,
                                                     const Eigen::Vector3f& P2,
                                                     const Eigen::Vector3f& P3,
                                                     float r1, float r2,
                                                     float r3) {
    // 初始化原点和向量p2、p3
    Eigen::Vector3f p1(0, 0, 0);
    Eigen::Vector3f p2 = P2 - P1;
    Eigen::Vector3f p3 = P3 - P1;

    // 计算基向量v1和v2
    Eigen::Vector3f v1 = p2 - p1;
    Eigen::Vector3f v2 = p3 - p1;

    // 计算正交坐标系的Xn、Yn、Zn轴
    Eigen::Vector3f Xn = v1.normalized();
    Eigen::Vector3f Zn = v1.cross(v2).normalized();
    Eigen::Vector3f Yn = Xn.cross(Zn);

    // 计算定位方程的中间变量
    float i = Xn.dot(v2);
    float d = Xn.dot(v1);
    float j = Yn.dot(v2);

    // 解定位方程，计算X、Y坐标
    float X = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
    float Y = (((r1 * r1) - (r3 * r3) + (i * i) + (j * j)) / (2 * j)) -
              ((i / j) * (X));

    // 计算Z坐标，得到两个可能的结果
    float Z1 = std::sqrt(std::max(0.0f, r1 * r1 - X * X - Y * Y));
    float Z2 = -Z1;

    // 根据计算得到的坐标，构造两个可能的结果向量
    Eigen::Vector3f result_0 = P1 + X * Xn + Y * Yn + Z1 * Zn;
    Eigen::Vector3f result_1 = P1 + X * Xn + Y * Yn + Z2 * Zn;

    // 返回第一个结果向量
    return result_0;
}

bool BladeModelTransformer::getIntersectionOfTwoCircles(
    const QVector2D& c0, const QVector2D& c1, float r0, float r1,
    std::vector<QVector2D>& result) {
    float x = c0.x();
    float y = c0.y();
    float R = r0;
    float a = c1.x();
    float b = c1.y();
    float S = r1;
    float d = sqrtf((a - x) * (a - x) + (b - y) * (b - y));

    if (d > (R + S) || d < (abs(R - S))) {
        printf("distance %f Two circles have no intersection", d);
        return false;
    } else if (d == 0) {
        printf("Two circles have same center!");
        return false;
        //        return None,None
    } else {
        float A   = (R * R - S * S + d * d) / (2 * d);
        float h   = sqrtf(R * R - A * A);
        float x2  = x + A * (a - x) / d;
        float y2  = y + A * (b - y) / d;
        float x3  = x2 - h * (b - y) / d;
        float y3  = y2 + h * (a - x) / d;
        float x4  = x2 + h * (b - y) / d;
        float y4  = y2 - h * (a - x) / d;
        result[0] = QVector2D(x3, y3);
        result[1] = QVector2D(x4, y4);
        return true;
    }
}

BladeModelTransformer::BladeModelTransformer(
    const QVector3D& _p_A, const QVector3D& _p_D, const QVector3D& _p_F,
    const QVector3D& _p_G, float _l_AE, float _l_AB, float _l_BC, float _l_AC,
    const QVector3D& _p_2_nC, const QVector3D& _p_2_nA, float _l_nBnC)
    : p_A(_p_A), p_D(_p_D), p_F(_p_F), p_G(_p_G), l_AE(_l_AE), l_AB(_l_AB),
      l_BC(_l_BC), l_AC(_l_AC), p_2_nC(_p_2_nC), p_2_nA(_p_2_nA),
      l_nBnC(_l_nBnC) { }

/**
 * 类构造函数：BladeModelTransformer
 * 该构造函数负责初始化BladeModelTransformer类的实例。
 * 该过程主要包括读取模型配置、初始化坐标变换矩阵，并加载静态模型组件。
 *
 * 参数：无
 * 返回值：无
 */
BladeModelTransformer::BladeModelTransformer() {
    // 从配置中读取模型类型和基础路径
    std::string model_type = config.model_type;
    std::filesystem::path model_base_path(config.blade_model_path);

    // 初始化坐标系矩阵
    Eigen::Matrix4f yuanpan_w_coord      = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f chandaobeimian_coord = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f global_coord         = Eigen::Matrix4f::Identity();

    // 设置全局坐标变换，主要用于调整模型的朝向和位置
    {
        // 定义并设置旋转矩阵，用于调整模型的初始朝向
        Eigen::Matrix3f R;
        R << 0, 1, 0, -1, 0, 0, 0, 0, 1;
        global_coord.topLeftCorner(3, 3)  = R;
        global_coord.topRightCorner(3, 1) = A;

        // 加载四个静态模型组件，应用全局坐标变换
        // 这部分代码通过基础路径和模型类型动态构建每个静态模型的文件路径，
        // 然后使用加载模型函数加载这些模型，并应用前面定义的全局坐标变换。
        std::filesystem::path static_mesh_path(model_base_path);
        static_mesh_path.append("4.1" + model_type);
        static_mesh_1.loadModel(static_mesh_path,
                                global_coord.topLeftCorner(3, 3),
                                global_coord.topRightCorner(3, 1));

        std::filesystem::path static_mesh_path_2(model_base_path);
        static_mesh_path_2.append("4.2" + model_type);
        static_mesh_2.loadModel(static_mesh_path_2,
                                global_coord.topLeftCorner(3, 3),
                                global_coord.topRightCorner(3, 1));

        std::filesystem::path static_mesh_path_3(model_base_path);
        static_mesh_path_3.append("4.3" + model_type);
        static_mesh_3.loadModel(static_mesh_path_3,
                                global_coord.topLeftCorner(3, 3),
                                global_coord.topRightCorner(3, 1));

        std::filesystem::path static_mesh_path_4(model_base_path);
        static_mesh_path_4.append("4.4" + model_type);
        static_mesh_4.loadModel(static_mesh_path_4,
                                global_coord.topLeftCorner(3, 3),
                                global_coord.topRightCorner(3, 1));
    }

    // 根据给定的参数配置并加载模型
    {
        // 计算从点A到B和从点A到C的单位向量
        Eigen::Vector3f AB = (init_B - A).normalized();
        Eigen::Vector3f AC = (init_C - A).normalized();

        // 构建旋转矩阵R，使其第一列是向量AC，第二列是AC和AB的叉积的单位向量，第三列是第二列和第一列的叉积的单位向量
        Eigen::Matrix3f R;
        R.col(0) = AC;
        R.col(2) = AC.cross(AB).normalized();
        R.col(1) = R.col(2).cross(R.col(0)).normalized();

        // 将旋转矩阵和位移向量A存储到qianyinjia_world中
        qianyinjia_world.topLeftCorner(3, 3)  = R;
        qianyinjia_world.topRightCorner(3, 1) = A;

        // 构建模型路径并加载模型
        std::filesystem::path qianyinjia_mesh_path(model_base_path);
        qianyinjia_mesh_path.append("1.2" + model_type);
        qianyinjia_mesh.loadModel(qianyinjia_mesh_path,
                                  qianyinjia_world.topLeftCorner(3, 3),
                                  qianyinjia_world.topRightCorner(3, 1));

        // 计算模型在全局坐标系中的位置
        qianyinjia_local = global_coord.inverse() * qianyinjia_world;
    }

    // 配置并加载油缸模型，以油缸的长轴为x轴，短轴为y轴
    // 计算并标准化油缸一y轴的起始和结束向量
    Eigen::Vector3f C_y_axis_start(8345.486 / 10000.0, -8247.5 / 10000.0,
                                   5428.826 / 10000.0);
    Eigen::Vector3f C_y_axis_end(6395.486 / 10000.0, -8247.5 / 10000.0,
                                 5428.826 / 10000.0);
    // 油缸一的y轴向量配置
    {
        yougang1_y_axis = (C_y_axis_end - C_y_axis_start).normalized();
        yougang1_y_axis = transform(global_coord.inverse(), yougang1_y_axis);

        // 构建油缸一的下部模型的坐标系
        Eigen::Vector3f x_start = init_C;
        Eigen::Vector3f x_end   = G;
        Eigen::Vector3f y_start = C_y_axis_start;
        Eigen::Vector3f y_end   = C_y_axis_end;

        Eigen::Matrix4f temp            = Eigen::Matrix4f::Identity();
        temp.topLeftCorner(3, 3).col(0) = (x_end - x_start).normalized();
        temp.topLeftCorner(3, 3).col(1) =
            ((y_end - y_start) -
             (y_end - y_start).dot((x_end - x_start).normalized()) *
                 (x_end - x_start).normalized())
                .normalized();
        temp.topLeftCorner(3, 3).col(2) =
            (x_end - x_start)
                .normalized()
                .cross((y_end - y_start).normalized());
        temp.topRightCorner(3, 1) = init_C;

        // 加载油缸一下部的模型
        std::filesystem::path yougang1_down_mesh_path(model_base_path);
        yougang1_down_mesh_path.append("6.2" + model_type);
        yougang1_down_mesh.loadModel(yougang1_down_mesh_path,
                                     temp.topLeftCorner(3, 3),
                                     temp.topRightCorner(3, 1));
    }

    // 配置并加载油缸一的上部模型
    {
        // 构建油缸一的上部模型的坐标系
        Eigen::Vector3f x_start = G;
        Eigen::Vector3f x_end   = init_C;
        Eigen::Vector3f y_start = C_y_axis_start;
        Eigen::Vector3f y_end   = C_y_axis_end;

        Eigen::Matrix4f temp            = Eigen::Matrix4f::Identity();
        temp.topLeftCorner(3, 3).col(0) = (x_end - x_start).normalized();
        temp.topLeftCorner(3, 3).col(1) =
            ((y_end - y_start) -
             (y_end - y_start).dot((x_end - x_start).normalized()) *
                 (x_end - x_start).normalized())
                .normalized();
        temp.topLeftCorner(3, 3).col(2) =
            (x_end - x_start)
                .normalized()
                .cross((y_end - y_start).normalized());
        temp.topRightCorner(3, 1) = G;

        // 加载油缸一上部的模型
        std::filesystem::path yougang1_up_mesh_path(model_base_path);
        yougang1_up_mesh_path.append("6.1" + model_type);
        yougang1_up_mesh.loadModel(yougang1_up_mesh_path,
                                   temp.topLeftCorner(3, 3),
                                   temp.topRightCorner(3, 1));
    }
    // 定义油缸二的Y轴起始和结束位置
    Eigen::Vector3f B_y_axis_start(9795.486 / 10000.0, 8252.5 / 10000.0,
                                   5428.826 / 10000.0);
    Eigen::Vector3f B_y_axis_end(6395.486 / 10000.0, 8252.5 / 10000.0,
                                 5428.826 / 10000.0);
    {
        // 计算并设置油缸二Y轴方向向量，考虑了全局坐标变换
        yougang2_y_axis = (B_y_axis_end - B_y_axis_start).normalized();
        yougang2_y_axis = transform(global_coord.inverse(), yougang2_y_axis);

        // 定义坐标系重构所需起点和终点
        Eigen::Vector3f x_start = init_B;
        Eigen::Vector3f x_end   = D;
        Eigen::Vector3f y_start = B_y_axis_start;
        Eigen::Vector3f y_end   = B_y_axis_end;

        // 构建坐标系，用于加载模型
        Eigen::Matrix4f temp            = Eigen::Matrix4f::Identity();
        temp.topLeftCorner(3, 3).col(0) = (x_end - x_start).normalized();
        temp.topLeftCorner(3, 3).col(1) =
            ((y_end - y_start) -
             (y_end - y_start).dot((x_end - x_start).normalized()) *
                 (x_end - x_start).normalized())
                .normalized();
        temp.topLeftCorner(3, 3).col(2) =
            (x_end - x_start)
                .normalized()
                .cross((y_end - y_start).normalized());
        temp.topRightCorner(3, 1) = init_B;

        // 加载油缸二下部模型
        std::filesystem::path yougang2_down_mesh_path(model_base_path);
        yougang2_down_mesh_path.append("7.2" + model_type);
        yougang2_down_mesh.loadModel(yougang2_down_mesh_path,
                                     temp.topLeftCorner(3, 3),
                                     temp.topRightCorner(3, 1));
    }
    {
        // 定义坐标系重构所需起点和终点，用于油缸二上部模型
        Eigen::Vector3f x_start = D;
        Eigen::Vector3f x_end   = init_B;
        Eigen::Vector3f y_start = B_y_axis_start;
        Eigen::Vector3f y_end   = B_y_axis_end;

        // 构建坐标系，用于加载模型
        Eigen::Matrix4f temp            = Eigen::Matrix4f::Identity();
        temp.topLeftCorner(3, 3).col(0) = (x_end - x_start).normalized();
        temp.topLeftCorner(3, 3).col(1) =
            ((y_end - y_start) -
             (y_end - y_start).dot((x_end - x_start).normalized()) *
                 (x_end - x_start).normalized())
                .normalized();
        temp.topLeftCorner(3, 3).col(2) =
            (x_end - x_start)
                .normalized()
                .cross((y_end - y_start).normalized());
        temp.topRightCorner(3, 1) = D;

        // 加载油缸二上部模型
        std::filesystem::path yougang2_up_mesh_path(model_base_path);
        yougang2_up_mesh_path.append("7.1" + model_type);
        yougang2_up_mesh.loadModel(yougang2_up_mesh_path,
                                   temp.topLeftCorner(3, 3),
                                   temp.topRightCorner(3, 1));
    }
    // 定义油缸三的Y轴起始和结束位置
    Eigen::Vector3f E_y_axis_start(9795.486 / 10000.0, 8252.5 / 10000.0,
                                   5428.826 / 10000.0);
    Eigen::Vector3f E_y_axis_end(6395.486 / 10000.0, 8252.5 / 10000.0,
                                 5428.826 / 10000.0);

    // 计算并设置油缸三Y轴方向向量，考虑了全局坐标变换
    {
        yougang3_y_axis = (E_y_axis_end - E_y_axis_start).normalized();
        yougang3_y_axis = transform(global_coord.inverse(), yougang3_y_axis);

        // 初始化用于构建变换矩阵的起点和终点
        Eigen::Vector3f x_start = init_E;
        Eigen::Vector3f x_end   = F;
        Eigen::Vector3f y_start = E_y_axis_start;
        Eigen::Vector3f y_end   = E_y_axis_end;

        // 构建变换矩阵，用于加载模型
        Eigen::Matrix4f temp            = Eigen::Matrix4f::Identity();
        temp.topLeftCorner(3, 3).col(0) = (x_end - x_start).normalized();
        temp.topLeftCorner(3, 3).col(1) =
            ((y_end - y_start) -
             (y_end - y_start).dot((x_end - x_start).normalized()) *
                 (x_end - x_start).normalized())
                .normalized();
        temp.topLeftCorner(3, 3).col(2) =
            (x_end - x_start)
                .normalized()
                .cross((y_end - y_start).normalized());
        temp.topRightCorner(3, 1) = init_E;

        // 加载油缸下部模型
        std::filesystem::path yougang3_down_mesh_path(model_base_path);
        yougang3_down_mesh_path.append("5.2" + model_type);
        yougang3_down_mesh.loadModel(yougang3_down_mesh_path,
                                     temp.topLeftCorner(3, 3),
                                     temp.topRightCorner(3, 1));
    }
    // 重复上述过程，但此次是为了加载油缸上部的模型
    {
        Eigen::Vector3f x_start = init_E;
        Eigen::Vector3f x_end   = F;
        Eigen::Vector3f y_start = E_y_axis_start;
        Eigen::Vector3f y_end   = E_y_axis_end;

        Eigen::Matrix4f temp            = Eigen::Matrix4f::Identity();
        temp.topLeftCorner(3, 3).col(0) = (x_end - x_start).normalized();
        temp.topLeftCorner(3, 3).col(1) =
            ((y_end - y_start) -
             (y_end - y_start).dot((x_end - x_start).normalized()) *
                 (x_end - x_start).normalized())
                .normalized();
        temp.topLeftCorner(3, 3).col(2) =
            (x_end - x_start)
                .normalized()
                .cross((y_end - y_start).normalized());
        temp.topRightCorner(3, 1) = F;

        // 加载油缸上部模型
        std::filesystem::path yougang3_up_mesh_path(model_base_path);
        yougang3_up_mesh_path.append("5.1" + model_type);
        yougang3_up_mesh.loadModel(yougang3_up_mesh_path,
                                   temp.topLeftCorner(3, 3),
                                   temp.topRightCorner(3, 1));
    }

    // 将点A、D、G、F、init_E、init_B、init_C从全局坐标系转换为局部坐标系
    A      = transform(global_coord.inverse(), A);
    D      = transform(global_coord.inverse(), D);
    G      = transform(global_coord.inverse(), G);
    F      = transform(global_coord.inverse(), F);
    init_E = transform(global_coord.inverse(), init_E);
    init_B = transform(global_coord.inverse(), init_B);
    init_C = transform(global_coord.inverse(), init_C);

    // 设置（圆盘）模型的世界坐标系并加载模型
    {
        Eigen::Vector3f center = H;        // 设置圆盘中心点
        Eigen::Matrix3f R;                 // 旋转矩阵初始化
        R << 0, 1, 0, -1, 0, 0, 0, 0, 1;   // 设定旋转矩阵
        yuanpan_world.topLeftCorner(3, 3) = R;   // 应用旋转部分到世界坐标系
        yuanpan_world.topRightCorner(3, 1) =
            center;   // 应用平移部分到世界坐标系
        std::filesystem::path yuanpan_mesh_path(
            model_base_path);   // 拼接模型文件路径
        yuanpan_mesh_path.append("1.1" +
                                 model_type);   // 添加模型版本和类型后缀
        yuanpan_mesh.loadModel(yuanpan_mesh_path,   // 加载模型
                               yuanpan_world.topLeftCorner(3, 3),
                               yuanpan_world.topRightCorner(3, 1));
        init_yuanpan_local =
            qianyinjia_world.inverse() *
            yuanpan_world;   // 计算圆盘在牵引架局部坐标系中的位置
    }

    // 设置（铲刀支架）模型的世界坐标系并加载模型
    {
        Eigen::Vector3f left_K(1313.574 / 1000.0, -822.781 / 1000.0,
                               -437.801 / 1000.0);   // 设置左K点坐标

        Eigen::Matrix3f R;                      // 旋转矩阵初始化
        R.col(2) = (K - left_K).normalized();   // 根据K和left_K点确定z轴方向
        R.col(0) = (init_J - K).normalized();   // 根据init_J和K点确定x轴方向
        R.col(1) =
            R.col(2).normalized().cross(R.col(0).normalized());   // 计算y轴方向
        chandaozhijia_world.topLeftCorner(3, 3) =
            R;   // 应用旋转部分到世界坐标系
        chandaozhijia_world.topRightCorner(3, 1) =
            K;   // 应用平移部分到世界坐标系
        std::filesystem::path chandaozhijia_mesh_path(
            model_base_path);   // 拼接模型文件路径
        chandaozhijia_mesh_path.append("2.2" +
                                       model_type);   // 添加模型版本和类型后缀

        chandaozhijia_mesh.loadModel(chandaozhijia_mesh_path,   // 加载模型
                                     chandaozhijia_world.topLeftCorner(3, 3),
                                     chandaozhijia_world.topRightCorner(3, 1));
        chandaozhijia_local =
            yuanpan_world.inverse() *
            chandaozhijia_world;   // 计算铲刀支架在圆盘局部坐标系中的位置
    }
    // 将坐标系从圆盘世界坐标系转换到当前坐标系
    I      = transform(yuanpan_world.inverse(), I);
    init_J = transform(yuanpan_world.inverse(), init_J);
    K      = transform(yuanpan_world.inverse(), K);

    {
        // 初始化旋转矩阵R，用于调整模型朝向
        Eigen::Matrix3f R;
        // 根据当前位姿L和初始位姿init_M计算R的第一列向量，并归一化
        R.col(0) = (L - init_M).normalized();
        // 使用第一列向量和单位向量X轴的叉乘得到第二列向量，并归一化
        R.col(1) = R.col(0).normalized().cross(Eigen::Vector3f(1, 0, 0));
        // 使用第一列和第二列向量的叉乘得到第三列向量，并归一化
        R.col(2) = R.col(0).normalized().cross(R.col(1).normalized());
        // 将旋转矩阵R的前3x3部分赋值给chandao_world矩阵
        chandao_world.topLeftCorner(3, 3) = R;
        // 将初始位姿init_M赋值给chandao_world矩阵的右上角3x1部分
        chandao_world.topRightCorner(3, 1) = init_M;
        // 构建模型路径
        std::filesystem::path chandao_mesh_path(model_base_path);
        chandao_mesh_path.append("2.1" + model_type);
        // 加载模型，使用更新后的世界矩阵
        chandao_mesh.loadModel(chandao_mesh_path,
                               chandao_world.topLeftCorner(3, 3),
                               chandao_world.topRightCorner(3, 1));
        // 将chandao_left和chandao_right从毫米转换为米，并根据旋转矩阵R和初始位姿调整坐标
        chandao_left /= 1000;
        chandao_right /= 1000;
        chandao_left  = R.transpose() * (chandao_left - init_M);
        chandao_right = R.transpose() * (chandao_right - init_M);

        // 计算局部坐标系下的位姿
        chandao_local = chandaozhijia_world.inverse() * chandao_world;
    }

    // 将L和init_M从全局坐标系转换到chandaozhijia_world的局部坐标系
    L      = transform(chandaozhijia_world.inverse(), L);
    init_M = transform(chandaozhijia_world.inverse(), init_M);
}

/**
 * 更新Blade模型的当前姿态。
 *
 * @param l_CG 圆心G到圆心C的距离。
 * @param l_BD 圆心B到圆心D的距离。
 * @param l_EF 圆心E到圆心F的距离。
 * @param rotation 关于y轴的旋转角度。
 * @param l_IJ 圆心I到圆心J的距离。
 * @param l_LM 圆心L到点M的距离。
 */
void BladeModelTransformer::setCurrentPoseNew(float l_CG, float l_BD,
                                              float l_EF, float rotation,
                                              float l_IJ, float l_LM) {
    // 计算向量长度和缩放因子
    float l_AE  = (A - init_E).norm();
    float l_AB  = (A - init_B).norm();
    float sigma = l_AE / l_AB;
    l_EF /= sigma;
    Eigen::Vector3f new_F = F / sigma;

    // 通过三边测量计算点B的位置
    Eigen::Vector3f B = trilateration(A, D, new_F, l_AB, l_BD, l_EF);

    // 计算点C的位置
    float l_AC        = (A - init_C).norm();
    float l_BC        = (init_B - init_C).norm();
    Eigen::Vector3f C = trilateration(A, B, G, l_AC, l_BC, l_CG);

    // 计算局部坐标系的x, y, z轴
    Eigen::Vector3f AB_w  = B - A;
    Eigen::Vector3f AC_w  = C - A;
    Eigen::Vector3f z_vec = AC_w.cross(AB_w).normalized();
    Eigen::Vector3f x_vec = AC_w.normalized();
    Eigen::Vector3f y_vec = z_vec.cross(x_vec).normalized();
    qianyinjia_local_estimated.topLeftCorner(3, 3).col(0) = x_vec;
    qianyinjia_local_estimated.topLeftCorner(3, 3).col(1) = y_vec;
    qianyinjia_local_estimated.topLeftCorner(3, 3).col(2) = z_vec;
    qianyinjia_local_estimated.topRightCorner(3, 1)       = A;

    // 更新模型的局部坐标系和全局坐标系
    Eigen::Vector3f E =
        transform(qianyinjia_local_estimated,
                  transform(qianyinjia_local.inverse(), init_E));
    B = transform(qianyinjia_local_estimated,
                  transform(qianyinjia_local.inverse(), init_B));
    C = transform(qianyinjia_local_estimated,
                  transform(qianyinjia_local.inverse(), init_C));

    // 设置Blade模型的模型矩阵
    qianyinjia_mesh.setModelMatrix(qianyinjia_local_estimated);

    // 更新yougang1模型的坐标系和模型矩阵
    Eigen::Vector3f col_0        = (G - C).normalized();
    Eigen::Vector3f col_1        = yougang1_y_axis.normalized();
    Eigen::Matrix4f T            = Eigen::Matrix4f::Identity();
    T.topLeftCorner(3, 3).col(0) = col_0.normalized();
    T.topLeftCorner(3, 3).col(1) = col_1.normalized();
    T.topLeftCorner(3, 3).col(2) = col_0.normalized().cross(col_1.normalized());
    T.topRightCorner(3, 1)       = G;
    yougang1_up_mesh.setModelMatrix(T);
    T.topRightCorner(3, 1) = C;
    yougang1_down_mesh.setModelMatrix(T);

    // 更新yougang2模型的坐标系和模型矩阵
    col_0                        = (D - B).normalized();
    col_1                        = yougang2_y_axis.normalized();
    T.topLeftCorner(3, 3).col(0) = col_0.normalized();
    T.topLeftCorner(3, 3).col(1) = col_1.normalized();
    T.topLeftCorner(3, 3).col(2) = col_0.normalized().cross(col_1.normalized());
    T.topRightCorner(3, 1)       = D;
    yougang2_up_mesh.setModelMatrix(T);
    T.topRightCorner(3, 1) = B;
    yougang2_down_mesh.setModelMatrix(T);

    // 更新yougang3模型的坐标系和模型矩阵
    col_0                        = (F - E).normalized();
    col_1                        = yougang3_y_axis.normalized();
    T.topLeftCorner(3, 3).col(0) = col_0.normalized();
    T.topLeftCorner(3, 3).col(1) = col_1.normalized();
    T.topLeftCorner(3, 3).col(2) = col_0.normalized().cross(col_1.normalized());
    T.topRightCorner(3, 1)       = F;
    yougang3_up_mesh.setModelMatrix(T);
    T.topRightCorner(3, 1) = E;
    yougang3_down_mesh.setModelMatrix(T);

    // 更新圆盘的旋转和模型矩阵
    float yuanpan_rotation = rotation;
    Eigen::Matrix4f temp   = Eigen::Matrix4f::Identity();
    temp.topLeftCorner(3, 3) =
        Eigen::AngleAxisf(yuanpan_rotation, -Eigen::Vector3f::UnitZ()).matrix();
    Eigen::Matrix4f yuanpan_local = init_yuanpan_local * temp;
    yuanpan_world                 = qianyinjia_local_estimated * yuanpan_local;
    yuanpan_mesh.setModelMatrix(yuanpan_world);

    // 计算并更新铲刀模型的坐标系和模型矩阵
    float l_JK = (init_J - K).norm();
    std::vector<Eigen::Vector2f> res(2);
    bool flag = intersection_two_circles(Eigen::Vector2f(I.y(), I.z()),
                                         Eigen::Vector2f(K.y(), K.z()), l_IJ,
                                         l_JK, res);
    if (!flag) {
        std::cout << "no intersection, reset" << std::endl;
    }
    // 选择正确的交点
    if (res[0].x() > res[1].x())
        res[0] = res[1];
    Eigen::Vector3f J = Eigen::Vector3f(I.x(), res[0].x(), res[0].y());

    Eigen::Matrix3f R = chandaozhijia_local.topLeftCorner(3, 3);
    R.col(0)          = (J - K).normalized();
    R.col(1)          = R.col(2).normalized().cross(R.col(0).normalized());
    chandaozhijia_local.topLeftCorner(3, 3) = R;
    chandaozhijia_world =
        qianyinjia_local_estimated * yuanpan_local * chandaozhijia_local;
    chandaozhijia_mesh.setModelMatrix(chandaozhijia_world);

    // 更新铲刀终点M的位置
    Eigen::Vector3f M                  = (init_M - L).normalized() * l_LM + L;
    chandao_local.topRightCorner(3, 1) = M;
    chandao_world = qianyinjia_local_estimated * yuanpan_local *
                    chandaozhijia_local * chandao_local;
    chandao_mesh.setModelMatrix(chandao_world);

    // 更新铲刀左右两侧点的Z坐标
    Vertex left_vertex = chandao_mesh.getVertex(config.chandao_left_vertex_idx);
    Vertex right_vertex =
        chandao_mesh.getVertex(config.chandao_right_vertex_idx);

    chandao_left_z  = getWorldZ(chandao_world, left_vertex);
    chandao_right_z = getWorldZ(chandao_world, right_vertex);

    // 初始化或更新铲刀设定的左右Z坐标
    if (chandao_set_left_z < -1e4)
        chandao_set_left_z = chandao_left_z;
    if (chandao_set_right_z < -1e4)
        chandao_set_right_z = chandao_right_z;

    // 更新当前铲刀左右两侧的实际位置
    cur_chandao_left  = transform(chandao_world, chandao_left);
    cur_chandao_right = transform(chandao_world, chandao_right);
}

/**
 * 检测两个圆的交点。
 *
 * 此函数用于计算给定两个圆的交点。如果两个圆相交，将交点的坐标存储在结果向量中，并返回true。
 * 如果两个圆不相交，则返回false。特定情况，如两个圆重合或完全分离，也会被妥善处理。
 *
 * @param c0 第一个圆的中心点坐标（x, y）。
 * @param c1 第二个圆的中心点坐标（x, y）。
 * @param r0 第一个圆的半径。
 * @param r1 第二个圆的半径。
 * @param result 存储两个圆交点的向量。每个交点以Eigen::Vector2f的形式存储。
 * @return 如果两个圆相交，返回true；否则返回false。
 */
bool BladeModelTransformer::intersection_two_circles(
    const Eigen::Vector2f& c0, const Eigen::Vector2f& c1, float r0, float r1,
    std::vector<Eigen::Vector2f>& result) {
    // 计算两个圆心之间的距离
    float x = c0.x();
    float y = c0.y();
    float R = r0;
    float a = c1.x();
    float b = c1.y();
    float S = r1;
    float d = sqrtf((a - x) * (a - x) + (b - y) * (b - y));

    // 判断两个圆的位置关系：无交集、内含、相切或相交
    if (d > (R + S) || d < (abs(R - S))) {
        printf("distance %f Two circles have no intersection", d);
        return false;
    } else if (d == 0) {
        printf("Two circles have same center!");
        return false;
    } else {
        // 计算交点坐标
        float A  = (R * R - S * S + d * d) / (2 * d);
        float h  = sqrtf(R * R - A * A);
        float x2 = x + A * (a - x) / d;
        float y2 = y + A * (b - y) / d;
        float x3 = x2 - h * (b - y) / d;
        float y3 = y2 + h * (a - x) / d;
        float x4 = x2 + h * (b - y) / d;
        float y4 = y2 - h * (a - x) / d;

        // 存储交点结果
        result.resize(2);
        result[0] = Eigen::Vector2f(x3, y3);
        result[1] = Eigen::Vector2f(x4, y4);
        return true;
    }
}

/**
 * 获取最低的Z值
 * 
 * 本函数用于获取BladeModelTransformer对象中最低的Z值。当前实现中，由于没有实际计算和检索过程，
 * 所以直接返回固定值0。在未来的开发中，这个函数可能会根据实际的模型数据进行计算。
 * 
 * @return int 返回最低的Z值。当前版本直接返回0。
 */
int BladeModelTransformer::getLowestZ() {
    return 0;
}

/**
 * 获取顶点在世界坐标系中的Z值。
 * @param trans 一个4x4的变换矩阵，用于将顶点从局部坐标系转换到世界坐标系。
 * @param vertex 需要获取Z值的顶点，顶点包含一个位置属性（x, y, z）。
 * @return 顶点在世界坐标系中的Z值。
 */
float getWorldZ(const Eigen::Matrix4f trans, const Vertex& vertex) {
    // 将顶点位置向量转换为齐次坐标表示，并应用变换矩阵。
    Eigen::Vector4f pos = Eigen::Vector4f(
        vertex.position.x(), vertex.position.y(), vertex.position.z(), 1.0);
    pos = trans * pos;

    // 通过除以w分量来标准化变换后的向量，得到在世界坐标系中的位置。
    pos /= pos.w();

    // 返回顶点在世界坐标系中的Z值。
    return pos.z();
}
