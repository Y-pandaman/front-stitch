#pragma once

#include <thrust/device_vector.h>

class GNSolver;

/**
 * @class Constraint
 * 这是一个约束类，用于定义和管理求解过程中的各种约束条件。
 * 它被设计为一个基类，通过派生实现具体的约束条件。
 */
class Constraint {
public:
    // GNSolver类的友元类，以便GNSolver可以访问Constraint的私有和保护成员。
    friend class GNSolver;

    /**
     * 构造函数。
     * 初始化gn_solver_为nullptr，weight_和row_、col_为0。
     */
    Constraint() : gn_solver_ {nullptr}, weight_ {0}, row_ {0}, col_ {0} {};

    /**
     * 虚拟析构函数。
     * 保证在派生类对象被删除时，能够正确调用派生类的析构函数。
     */
    virtual ~Constraint() { }

    /**
     * 获取约束类型的字符串表示。
     * @return 返回一个指向约束类型名称的字符指针。
     */
    virtual char* Ctype() {
        return "constraint_base";
    }

    /**
     * 初始化约束，接受一个GNSolver指针和权重参数。
     * 此函数为纯虚函数，需要在派生类中实现具体的初始化逻辑。
     * @param gn_solver 指向GNSolver对象的指针，用于约束与求解器的交互。
     * @param weight 约束的权重参数。
     * @return 返回一个布尔值，表示初始化是否成功。
     */
    virtual bool Init(GNSolver* gn_solver, float weight) = 0;

    /**
     * 无需参数的初始化函数，为派生类提供另一种初始化方式。
     * 此函数为纯虚函数，需要在派生类中实现。
     * @return 返回一个布尔值，表示初始化是否成功。
     */
    virtual bool Init()                                  = 0;

    /**
     * 设置约束的权重。
     * @param weight 想要设置的约束权重。
     */
    virtual void SetWeight(float weight) {
        weight_ = (weight);
    }

    /**
     * 计算并返回约束函数的值。
     * 此函数为纯虚函数，需要在派生类中实现具体的计算逻辑。
     * @param d_x 输入参数，表示当前的解向量。
     * @return 返回约束函数的值。
     */
    virtual float Val(thrust::device_vector<float>& d_x) {
        return 0.0f;
    }

    /**
     * 计算并更新Jacobian矩阵J的转置乘以J和J的转置乘以b。
     * 此函数为纯虚函数，需要在派生类中实现具体的计算逻辑。
     * @param d_JTJ_a 输出参数，存储J的转置乘以J的结果。
     * @param d_JTJ_ia 输出参数，存储J的转置乘以J的稀疏索引。
     * @param d_JTb 输出参数，存储J的转置乘以b的结果。
     * @param d_x 输入参数，表示当前的解向量。
     */
    virtual void GetJTJAndJTb(float* d_JTJ_a, int* d_JTJ_ia, float* d_JTb,
                              float* d_x) = 0;

protected:
    /**
     * 计算并更新b向量。
     * 此函数为纯虚函数，需要在派生类中实现具体的计算逻辑。
     * @param d_x 输入参数，表示当前的解向量。
     */
    virtual void b(float* d_x) = 0;

    // 指向GNSolver对象的指针，用于约束与求解器的交互。
    GNSolver* gn_solver_;
    // 约束的行数和列数。
    int row_, col_;
    // 约束的权重。
    float weight_;
    // 存储b向量的设备向量。
    thrust::device_vector<float> d_b_;
};

class DataTermConstraint : public Constraint {
public:
    DataTermConstraint() { }
    ~DataTermConstraint() { }

    char* Ctype() override {
        return "data_term_constraint";
    }

    bool Init(GNSolver* gn_solver, float weight) override;
    bool Init() override;
    void GetJTJAndJTb(float* d_JTJ_a, int* d_JTJ_ia, float* d_JTb,
                      float* d_x) override;

protected:
    void b(float* x) override;

private:
    void DirectiveJTJ(float* d_JTJ_a, int* d_JTJ_ia);
    void DirectiveJTb(float* d_JTb);
};

class SmoothTermConstraint : public Constraint {
public:
    SmoothTermConstraint() { }
    ~SmoothTermConstraint() { }

    char* Ctype() override {
        return "smooth_term_constraint";
    }

    bool Init(GNSolver* gn_solver, float weight) override;
    bool Init() override;
    void GetJTJAndJTb(float* d_JTJ_a, int* d_JTJ_ia, float* d_JTb,
                      float* d_x) override;

protected:
    void b(float* x) override;

private:
    void DirectiveJTJ(float* d_JTJ_a, int* d_JTJ_ia);
    void DirectiveJTb(float* d_JTb);
};

class ZeroTermConstraint : public Constraint {
public:
    ZeroTermConstraint() { }
    ~ZeroTermConstraint() { }

    char* Ctype() override {
        return "zero_term_constraint";
    }

    bool Init(GNSolver* gn_solver, float weight) override;
    bool Init() override;
    void GetJTJAndJTb(float* d_JTJ_a, int* d_JTJ_ia, float* d_JTb,
                      float* d_x) override;

protected:
    void b(float* x) override;

private:
    void DirectiveJTJ(float* d_JTJ_a, int* d_JTJ_ia);
    void DirectiveJTb(float* d_JTb);
};
