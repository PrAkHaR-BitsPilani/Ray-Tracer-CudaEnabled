#pragma once
#include "rtweekend.h"

#include "hittable.h"

class xy_rect : public hittable {
public:
    __device__ xy_rect() {}

    __device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k,
        material* mat)
        : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {};

    __device__     virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, AxisAllignedBoundingBox& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Z
        // dimension a small amount.
        output_box = AxisAllignedBoundingBox(glm::vec3(x0, y0, k - 0.0001), glm::vec3(x1, y1, k + 0.0001));
        return true;
    }

    __device__ void check()
    {
        printf("Inside XY\n");
    }

public:
    material* mp;
    float x0, x1, y0, y1, k;
};

class xz_rect : public hittable {
public:
    __device__ xz_rect() {}

    __device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _k,
        material* mat)
        : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {};

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, AxisAllignedBoundingBox& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the Y
        // dimension a small amount.
        output_box = AxisAllignedBoundingBox(glm::vec3(x0, k - 0.0001, z0), glm::vec3(x1, k + 0.0001, z1));
        return true;
    }

    __device__ void check()
    {
        printf("Inside XZ\n");
    }

public:
    material* mp;
    float x0, x1, z0, z1, k;
};

class yz_rect : public hittable {
public:
    __device__ yz_rect() {}

    __device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _k,
        material* mat)
        : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {};

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, AxisAllignedBoundingBox& output_box) const override {
        // The bounding box must have non-zero width in each dimension, so pad the X
        // dimension a small amount.
        output_box = AxisAllignedBoundingBox(glm::vec3(k - 0.0001, y0, z0), glm::vec3(k + 0.0001, y1, z1));
        return true;
    }

    __device__ void check()
    {
        printf("Inside YZ\n");
    }

public:
    material* mp;
    float y0, y1, z0, z1, k;
};

__device__ bool xy_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.orig.z) / r.dir.z;
    if (t < t_min || t > t_max)
        return false;
    auto x = r.orig.x + t * r.dir.x;
    auto y = r.orig.y + t * r.dir.y;
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    auto outward_normal = glm::vec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.at(t);
    return true;
}

__device__ bool xz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.orig.y) / r.dir.y;
    if (t < t_min || t > t_max)
        return false;
    auto x = r.orig.x + t * r.dir.x;
    auto z = r.orig.z + t * r.dir.z;
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    auto outward_normal = glm::vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.at(t);
    return true;
}

__device__ bool yz_rect::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    auto t = (k - r.orig.x) / r.dir.x;
    if (t < t_min || t > t_max)
        return false;
    auto y = r.orig.y + t * r.dir.y;
    auto z = r.orig.z + t * r.dir.z;
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    rec.u = (y - y0) / (y1 - y0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    auto outward_normal = glm::vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.p = r.at(t);
    return true;
}