#pragma once

#include "rtweekend.cuh"
#include "AxisAllignedBoundingBox.cuh"


class material;

struct hit_record {
    glm::vec3 p;
    glm::vec3 normal;
    material* mat_ptr;
    float t;   
    float u;
    float v;
    bool front_face;

    __device__ inline void set_face_normal(const ray& r, const glm::vec3& outward_normal) {
        front_face = glm::dot(r.dir, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};


class hittable {
public:
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;

    __device__ virtual bool bounding_box(float time0, float time1, AxisAllignedBoundingBox& outbox) const = 0;
};