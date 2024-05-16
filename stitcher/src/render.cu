#include "cuda_utils.h"
#include "render.cuh"

static inline __device__ bool RayCylinderIntersection_v2(float3 origin,
                                                         float3 dir, float r,
                                                         float3& intersection) {
    float3 abs_dir = fabs(dir);

    if (abs_dir.x < CLOSE_ZERO) {
        intersection.x = origin.x;
        intersection.z = sqrtf(powf(r, 2.0f) - powf(intersection.x, 2.0f));

        if ((intersection.z - origin.z) / dir.z < 0)
            intersection.z *= -1.0f;

    } else if (abs_dir.z < CLOSE_ZERO) {
        intersection.z = origin.z;
        intersection.x = sqrtf(powf(r, 2.0f) - powf(intersection.z, 2.0f));

        if ((intersection.x - origin.x) / dir.x < 0)
            intersection.x *= -1.0f;

    } else {
        float k = dir.z / dir.x;
        float b = origin.z - k * origin.x;

        float check =
            powf(2.0f * k * b, 2.0f) - 4 * (1 + k * k) * (b * b - r * r);
        if (check < 0)
            return false;

        float item = sqrtf(check);

        float x1 = (-2.0f * k * b + item) / (2 * (1 + k * k));
        float x2 = (-2.0f * k * b - item) / (2 * (1 + k * k));

        float z1 = k * x1 + b;
        float z2 = k * x2 + b;

        float tan_theta = dir.y / sqrtf(dir.x * dir.x + dir.z * dir.z);

        bool cross_1 =
            (x1 - origin.x) / dir.x > 0 && (z1 - origin.z) / dir.z > 0;
        bool cross_2 =
            (x2 - origin.x) / dir.x > 0 && (z2 - origin.z) / dir.z > 0;

        if (cross_1 && cross_2) {
            float t1_xz =
                sqrtf(powf(x1 - origin.x, 2.0f) + powf(z1 - origin.z, 2.0f));
            float t2_xz =
                sqrtf(powf(x2 - origin.x, 2.0f) + powf(z2 - origin.z, 2.0f));
            if (t1_xz > t2_xz) {
                intersection.x = x1;
                intersection.z = z1;
                intersection.y = origin.y + t1_xz * tan_theta;
            } else {
                intersection.x = x2;
                intersection.z = z2;
                intersection.y = origin.y + t2_xz * tan_theta;
            }

        } else {
            if (cross_1) {
                intersection.x = x1;
                intersection.z = z1;
            } else {
                intersection.x = x2;
                intersection.z = z2;
            }
            float t_xz     = sqrtf(powf(intersection.x - origin.x, 2.0f) +
                               powf(intersection.z - origin.z, 2.0f));
            intersection.y = origin.y + t_xz * tan_theta;
        }
    }
    return true;
}

static inline __device__ __host__ uchar3 interpolate1D(uchar3 v1, uchar3 v2,
                                                       float x) {
    return make_uchar3((uchar)((float)v1.x * (1.0f - x) + (float)v2.x * x),
                       (uchar)((float)v1.y * (1.0f - x) + (float)v2.y * x),
                       (uchar)((float)v1.z * (1.0f - x) + (float)v2.z * x));
}

static inline __device__ __host__ uchar3 interpolate2D(uchar3 v1, uchar3 v2,
                                                       uchar3 v3, uchar3 v4,
                                                       float x, float y) {
    uchar3 s = interpolate1D(v1, v2, x);
    uchar3 t = interpolate1D(v3, v4, x);
    return interpolate1D(s, t, y);
}

static inline __device__ uchar3 Bilinear(uchar3* src, int h, int w,
                                         float2 pixel) {
    int x0  = (int)pixel.x;
    int y0  = (int)pixel.y;
    int x1  = x0 + 1;
    int y1  = y0 + 1;
    float x = pixel.x - x0;
    float y = pixel.y - y0;

    int idx00 = x0 + y0 * w;
    int idx01 = idx00 + w;
    int idx10 = idx00 + 1;
    int idx11 = idx01 + 1;

    // Condition Boundary  [TODO]

    return interpolate2D(src[idx00], src[idx10], src[idx01], src[idx11], x, y);
}

__global__ void blend_extra_view_4channels_kernel(uchar3* dst_cyl_img,
                                                  uchar4* src_cyl_img,
                                                  int width, int height,
                                                  float w) {
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixelIdx < width * height) {
        float w_1               = (float)src_cyl_img[pixelIdx].w / 255.0;
        float w_2               = 1.0 - w_1;
        dst_cyl_img[pixelIdx].x = (w_1 * (float)src_cyl_img[pixelIdx].z +
                                   w_2 * (float)dst_cyl_img[pixelIdx].x);
        dst_cyl_img[pixelIdx].y = (w_1 * (float)src_cyl_img[pixelIdx].y +
                                   w_2 * (float)dst_cyl_img[pixelIdx].y);
        dst_cyl_img[pixelIdx].z = (w_1 * (float)src_cyl_img[pixelIdx].x +
                                   w_2 * (float)dst_cyl_img[pixelIdx].z);
    }
}

__global__ void blend_extra_view_kernel(uchar3* dst_cyl_img,
                                        uchar3* src_cyl_img, int width,
                                        int height, float w) {
    int pixelIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixelIdx < width * height) {
        if (src_cyl_img[pixelIdx].x > 0 || src_cyl_img[pixelIdx].y > 0 ||
            src_cyl_img[pixelIdx].z > 0) {
            float w_1 =
                max(src_cyl_img[pixelIdx].x,
                    max(src_cyl_img[pixelIdx].y, src_cyl_img[pixelIdx].z)) /
                255.0 * w;
            float w_2               = 1.0 - w_1;
            dst_cyl_img[pixelIdx].x = (w_1 * (float)src_cyl_img[pixelIdx].z +
                                       w_2 * (float)dst_cyl_img[pixelIdx].x);
            dst_cyl_img[pixelIdx].y = (w_1 * (float)src_cyl_img[pixelIdx].y +
                                       w_2 * (float)dst_cyl_img[pixelIdx].y);
            dst_cyl_img[pixelIdx].z = (w_1 * (float)src_cyl_img[pixelIdx].x +
                                       w_2 * (float)dst_cyl_img[pixelIdx].z);
        }
    }
}

__global__ void render_kernel(float4* intrin, float4* center_view_intrin,
                              float3* extrin_R, uchar3* cyl_img, int src_h,
                              int src_w, CylinderGPU cyl, float ox, float oy,
                              float oz, uchar3* dst, int idx,   // idx = 0, 1, 2
                              int height, int width, int view_num,
                              bool is_center_view) {
    int pixelIdx     = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread = blockDim.x * gridDim.x;
    int totalPixel   = height * width;

    float image_theta_range =
        (cyl.global_theta[1] - cyl.global_theta[0]) / view_num;
    float image_phi_range = cyl.global_phi[1] - cyl.global_phi[0];

    float theta_step = (cyl.global_theta[1] - cyl.global_theta[0]) / src_w;
    float phi_step   = image_phi_range / src_h;

    float center_theta = cyl.global_theta[0] + image_theta_range / 2.0f +
                         idx * image_theta_range;
    float center_phi = cyl.global_phi[0] + image_phi_range / 2.0f;
    float3 p_cyl = make_float3(sinf(center_theta) * cyl.r, center_phi * cyl.r,
                               cosf(center_theta) * cyl.r);
    float3 inter_loc = cyl.rotateVector_inv(p_cyl) + cyl.getCenter();

    float3 camera_center = make_float3(ox, oy, oz);
    float3 Y_axis =
        make_float3(cyl.rotation[3], cyl.rotation[4], cyl.rotation[5]);
    float3 Z_axis = normalize(inter_loc - camera_center);
    float3 X_axis = normalize(cross(Y_axis, Z_axis));
    Z_axis        = normalize(cross(X_axis, Y_axis));
    extrin_R[0]   = make_float3(X_axis.x, Y_axis.x, Z_axis.x);
    extrin_R[1]   = make_float3(X_axis.y, Y_axis.y, Z_axis.y);
    extrin_R[2]   = make_float3(X_axis.z, Y_axis.z, Z_axis.z);

    float3 p1 =
        make_float3(sinf(center_theta - image_theta_range / 2.0f) * cyl.r,
                    cyl.global_phi[0] * cyl.r,
                    cosf(center_theta - image_theta_range / 2.0f) * cyl.r);
    float3 p2 =
        make_float3(sinf(center_theta + image_theta_range / 2.0f) * cyl.r,
                    cyl.global_phi[1] * cyl.r,
                    cosf(center_theta + image_theta_range / 2.0f) * cyl.r);
    p1 = cyl.rotateVector_inv(p1) + cyl.getCenter() - camera_center;
    p2 = cyl.rotateVector_inv(p2) + cyl.getCenter() - camera_center;
    p1 = rotateVector(X_axis, Y_axis, Z_axis, p1);
    p2 = rotateVector(X_axis, Y_axis, Z_axis, p2);

    float fx, fy, cx, cy;
    fx = (1.0f - width) / (p1.x / p1.z - p2.x / p2.z);
    cx = -1.0f * p1.x / p1.z * fx;
    fy = (1.0f - height) / (p1.y / p1.z - p2.y / p2.z);
    cy = -1.0f * p1.y / p1.z * fy;
    if (!is_center_view) {
        fy = center_view_intrin->y;
        cy = center_view_intrin->w;
    }
    intrin->x = fx;
    intrin->y = fy;
    intrin->z = cx;
    intrin->w = cy;

    while (pixelIdx < totalPixel) {
        int x = pixelIdx % width;
        int y = pixelIdx / width;

        float3 dir = make_float3((x - cx) / fx, (y - cy) / fy, 1.0f);

        dir = rotateVector_inv(X_axis, Y_axis, Z_axis, dir);
        dir = cyl.rotateVector(normalize(dir));

        float3 origin = cyl.rotateVector(camera_center - cyl.getCenter());

        float3 P;
        bool success = RayCylinderIntersection_v2(origin, dir, cyl.r, P);

        if (success) {
            float theta = clamp(atan2f(P.x, P.z), -3.141592653 / 2 + 0.001,
                                3.141592653 / 2 - 0.001);
            float phi   = P.y / cyl.r;

            float u = (theta - cyl.global_theta[0]) / theta_step;
            float v = (phi - cyl.global_phi[0]) / phi_step;

            if (u >= 0 && v >= 0 && u < src_w - 1 && v < src_h - 1) {
                uchar3 color =
                    Bilinear(cyl_img, src_h, src_w, make_float2(u, v));
                dst[pixelIdx] = color;
            } else {
                dst[pixelIdx] = make_uchar3(0, 0, 0);
            }
        } else {
            dst[pixelIdx] = make_uchar3(0, 0, 0);
        }

        pixelIdx += total_thread;
    }
}

__host__ void BlendExtraViewToScreen_cuda(uchar3* dst_cyl_img,
                                          uchar3* src_cyl_img, int width,
                                          int height, float w) {
    int num_thread = 512;
    int num_block  = min(65535, (height * width + num_thread - 1) / num_thread);

    blend_extra_view_kernel<<<num_block, num_thread>>>(dst_cyl_img, src_cyl_img,
                                                       width, height, w);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

__host__ void BlendExtraViewToScreen4Channels_cuda(uchar3* dst_cyl_img,
                                                   uchar4* src_cyl_img,
                                                   int width, int height,
                                                   float w) {
    int num_thread = 512;
    int num_block  = min(65535, (height * width + num_thread - 1) / num_thread);

    blend_extra_view_4channels_kernel<<<num_block, num_thread>>>(
        dst_cyl_img, src_cyl_img, width, height, w);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

__host__ void RenderToScreen_cuda(std::vector<float4*>& novel_view_intrins,
                                  std::vector<float3*>& novel_view_extrin_Rs,
                                  CylinderImageGPU cyl_img,
                                  std::vector<uchar3*> novel_images,
                                  CylinderGPU cylinder, float3 origin,
                                  int novel_images_num) {
    if (novel_view_intrins.size() == 0) {
        for (int i = 0; i < novel_images_num; i++) {
            float4* temp_1;
            checkCudaErrors(cudaMalloc((void**)&temp_1, sizeof(float4)));
            novel_view_intrins.emplace_back(temp_1);
            float3* temp_2;
            checkCudaErrors(cudaMalloc((void**)&temp_2, sizeof(float3) * 3));
            novel_view_extrin_Rs.emplace_back(temp_2);
        }
    }

    int num_thread = 512;
    int height = cyl_img.height, width = cyl_img.width / novel_images_num;
    int num_block = min(65535, (height * width + num_thread - 1) / num_thread);

    render_kernel<<<num_block, num_thread>>>(
        novel_view_intrins[novel_images_num / 2],
        novel_view_intrins[novel_images_num / 2],
        novel_view_extrin_Rs[novel_images_num / 2], cyl_img.image,
        cyl_img.height, cyl_img.width, cylinder, origin.x, origin.y, origin.z,
        novel_images[novel_images_num / 2], novel_images_num / 2, height, width,
        novel_images_num, true);
    for (int i = 0; i < novel_images_num; i++) {
        if (i != (novel_images_num / 2))
            render_kernel<<<num_block, num_thread>>>(
                novel_view_intrins[i], novel_view_intrins[novel_images_num / 2],
                novel_view_extrin_Rs[i], cyl_img.image, cyl_img.height,
                cyl_img.width, cylinder, origin.x, origin.y, origin.z,
                novel_images[i], i, height, width, novel_images_num, false);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}