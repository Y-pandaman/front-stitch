#pragma once

#include "util/math_utils.h"  // 上
#include "util/helper_cuda.h" // 下
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <string>
#include <thrust/extrema.h>

struct Plane {
    Eigen::Matrix<float, 3, 1> N;
    float d;

    Plane(Eigen::Matrix<float, 3, 1> _N, float _d) : N(_N), d(_d) { }
};

struct Cylinder {
    Eigen::Matrix<float, 3, 3> rotation;
    Eigen::Matrix<float, 3, 1> center;
    float r;

    Cylinder()
        : center(Eigen::Vector3f(0, 0, 0)), r(0.0f),
          rotation(Eigen::Matrix3f::Identity()) { }

    Cylinder(Eigen::Matrix<float, 3, 1> _center, float _r,
             Eigen::Matrix<float, 3, 3> _rotation)
        : center(_center), r(_r), rotation(_rotation) { }
};

struct Ray {
    Eigen::Matrix<float, 3, 1> origin;
    Eigen::Matrix<float, 3, 1> dir;

    Ray(Eigen::Matrix<float, 3, 1> _origin, Eigen::Matrix<float, 3, 1> _dir)
        : origin(_origin), dir(_dir) {
        this->dir.normalize();
    }

    Eigen::Matrix<float, 3, 1> getPoint(float t) {
        return this->origin + t * dir;
    }
};

static std::ostream& operator<<(std::ostream& out, const Ray& A) {
    out << "Ray o: " << A.origin.transpose()
        << "\tRay dir: " << A.dir.transpose() << "\n";
    return out;
}

struct PinholeCamera {
    Eigen::Matrix3f K;
    Eigen::Matrix3f R;
    Eigen::Matrix<float, 3, 1> T;
    Eigen::Matrix<float, 3, 1> C;
    Eigen::Matrix<float, 4, 4> extrin;

    PinholeCamera() { }

    PinholeCamera(Eigen::Matrix3f _K, Eigen::Matrix3f _R,
                  Eigen::Matrix<float, 3, 1> _T, bool c2w = false)
        : K(_K) {
        if (c2w == false) {
            this->R = _R;
            this->T = _T;
            this->C = -1.0f * R.transpose() * _T;
        } else {
            this->R = _R.transpose();
            this->C = _T;
            this->T = -1.0f * this->R * this->C;
        }
        extrin                      = Eigen::Matrix<float, 4, 4>::Identity();
        extrin.topLeftCorner(3, 3)  = R;
        extrin.topRightCorner(3, 1) = T;
    }

    void setPos(Eigen::Vector3f _C) {
        C                           = _C;
        T                           = -R.transpose() * C;
        extrin                      = Eigen::Matrix<float, 4, 4>::Identity();
        extrin.topLeftCorner(3, 3)  = R;
        extrin.topRightCorner(3, 1) = T;
    }

    void setAxisY(Eigen::Vector3f axis_y) {
        axis_y.normalize();
        R.transposeInPlace();
        R.col(1) = axis_y;
        R.col(2) = R.col(0).cross(R.col(1));
        R.col(2).normalize();
        R.transposeInPlace();

        extrin                      = Eigen::Matrix<float, 4, 4>::Identity();
        extrin.topLeftCorner(3, 3)  = R;
        extrin.topRightCorner(3, 1) = T;
    }

    void setAxisZ(Eigen::Vector3f axis_z) {
        axis_z.normalize();
        R.transposeInPlace();
        R.col(2) = axis_z;
        R.col(0) = R.col(1).cross(R.col(2));
        R.col(0).normalize();
        R.transposeInPlace();

        extrin                      = Eigen::Matrix<float, 4, 4>::Identity();
        extrin.topLeftCorner(3, 3)  = R;
        extrin.topRightCorner(3, 1) = T;
    }

    Ray getRay(int u, int v) {
        // input pixel  return ray
        Eigen::Matrix<float, 3, 1> pixel;
        pixel << (float)u, (float)v, 1;
        Eigen::Matrix<float, 3, 1> dir = R.transpose() * K.inverse() * pixel;
        dir.normalize();
        return Ray(this->C, dir);
    }

    Eigen::Matrix<float, 2, 1> projToPixel(Eigen::Matrix<float, 3, 1> p) {
        Eigen::Matrix<float, 3, 1> m = this->K * this->R * (p - this->C);
        if (m(2, 0) == 0) {
            std::cout << "Error Proj" << std::endl;
        }
        float x = m(0, 0) / m(2, 0);
        float y = m(1, 0) / m(2, 0);

        Eigen::Matrix<float, 2, 1> out;
        out << x, y;
        return out;
    }
};

struct View {
    cv::Mat image;
    cv::Mat mask;
    PinholeCamera camera;

    View() { }

    View(const View& v) {
        image  = v.image;
        camera = v.camera;
        mask   = v.mask;
    }

    View(cv::Mat _image, cv::Mat _mask, PinholeCamera _camera)
        : image(_image), mask(_mask), camera(_camera) { }
};

struct CylinderImage {
    cv::Mat image;
    cv::Mat mask;
    int2 min_uv;

    CylinderImage() { }

    CylinderImage(cv::Mat _image, cv::Mat _mask, int2 _min_uv)
        : image(_image), mask(_mask), min_uv(_min_uv) { }
};

class CylinderStitcher {
public:
    std::vector<std::vector<float4> >
    findCorrespondences(std::vector<View> views);

    CylinderImage projToCylinderImage(std::vector<View> views,
                                      Cylinder cylinder);

    void alignImagesCPU(std::vector<CylinderImage>& cylinder_images,
                        std::vector<std::vector<float4> > match_corrs);

    void alignImagesGPU(std::vector<CylinderImage>& cylinder_images,
                        std::vector<std::vector<float4> > match_corrs);

    void show(CylinderImage cyl_image, std::vector<View> views,
              Cylinder cylinder);

    void stitch(std::vector<View> views);

private:
    std::vector<View> views_;
    std::vector<CylinderImage> cylinder_images_;
    Cylinder cylinder_;
    double theta_step_, phi_step_, global_min_theta_, global_max_theta_,
        global_min_phi_, global_max_phi_;
};
