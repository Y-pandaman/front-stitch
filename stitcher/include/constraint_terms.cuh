#pragma once

#include <thrust/device_vector.h>

class GNSolver;

class Constraint {
public:
    friend class GNSolver;

    Constraint() : gn_solver_ {nullptr}, weight_ {0}, row_ {0}, col_ {0} {};

    virtual ~Constraint() { }

    virtual char* Ctype() {
        return "constraint_base";
    }

    virtual bool Init(GNSolver* gn_solver, float weight) = 0;
    virtual bool Init()                                  = 0;

    virtual void SetWeight(float weight) {
        weight_ = (weight);
    }

    virtual float Val(thrust::device_vector<float>& d_x) {
        return 0.0f;
    }

    virtual void GetJTJAndJTb(float* d_JTJ_a, int* d_JTJ_ia, float* d_JTb,
                              float* d_x) = 0;

protected:
    virtual void b(float* d_x) = 0;

    GNSolver* gn_solver_;
    int row_, col_;
    float weight_;
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
