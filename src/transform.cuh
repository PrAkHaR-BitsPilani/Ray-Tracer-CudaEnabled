#pragma once

#include "hittable.cuh"
#include "rtweekend.cuh"

class Translate : public hittable {
public:
    __device__ Translate(hittable* p, const glm::vec3& displacement) : ptr(p), offset(displacement) {}

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AxisAllignedBoundingBox& output_box) const;

    hittable* ptr;
    glm::vec3 offset;
};

__device__ bool Translate::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    ray moved_r(r.orig - offset, r.dir,r.tm);
    if (!ptr->hit(moved_r, t_min, t_max, rec))
        return false;

    rec.p += offset;
    rec.set_face_normal(moved_r, rec.normal);

    return true;
}

__device__ bool Translate::bounding_box(float t0, float t1, AxisAllignedBoundingBox& output_box) const {
    if (!ptr->bounding_box(t0, t1, output_box))
        return false;

    output_box = AxisAllignedBoundingBox(
        output_box.minimum + offset,
        output_box.maximum + offset);

    return true;
}

class RotateY : public hittable {
public:
    __device__ RotateY(hittable* p, float angle);

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AxisAllignedBoundingBox& output_box) const {
        output_box = bbox;
        return hasbox;
    }

    hittable* ptr;
    float sin_theta;
    float cos_theta;
    bool hasbox;
    AxisAllignedBoundingBox bbox;
};

__device__ RotateY::RotateY(hittable* p, float angle) : ptr(p) {
    auto radians = degrees_to_radians(angle);
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    hasbox = ptr->bounding_box(0, 1, bbox);

    glm::vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    glm::vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                auto x = i * bbox.maximum.x() + (1 - i) * bbox.minimum.x();
                auto y = j * bbox.maximum.y() + (1 - j) * bbox.minimum.y();
                auto z = k * bbox.maximum.z() + (1 - k) * bbox.minimum.z();

                auto newx = cos_theta * x + sin_theta * z;
                auto newz = -sin_theta * x + cos_theta * z;

                glm::vec3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    bbox = AxisAllignedBoundingBox(min, max);
}

__device__ bool RotateY::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    glm::vec3 origin = r.orig;
    glm::vec3 direction = r.dir;

    origin[0] = cos_theta * r.orig[0] - sin_theta * r.orig[2];
    origin[2] = sin_theta * r.orig[0] + cos_theta * r.orig[2];

    direction[0] = cos_theta * r.dir[0] - sin_theta * r.dir[2];
    direction[2] = sin_theta * r.dir[0] + cos_theta * r.dir[2];

    ray rotated_r(origin, direction, r.tm);

    if (!ptr->hit(rotated_r, t_min, t_max, rec))
        return false;

    glm::vec3 p = rec.p;
    glm::vec3 normal = rec.normal;

    p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
    p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

    normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
    normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

    rec.p = p;
    rec.set_face_normal(rotated_r, normal);

    return true;
}
