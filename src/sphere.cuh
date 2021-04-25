#pragma once

#include "rtweekend.cuh"

#include "hittable.cuh"


class sphere : public hittable {
public:
    __device__ sphere()
        : center(glm::vec3(0.0f)), radius(1.0f), mat_ptr(nullptr) {};

    __device__ sphere(glm::vec3 cen, float r, material* m)
        : center(cen), radius(r), mat_ptr(m) {};

    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, AxisAllignedBoundingBox& outbox) const override;

private:
    __device__ static void get_sphere_uv(const glm::vec3 p, float u, float v)
    {
        auto theta = acos(-p.y());
        auto phi = atan2(-p.z(), p.x()) + pi;

        u = phi / (2 * pi);
        v = theta / pi;
    }

public:
    glm::vec3 center;
    float radius;
    material* mat_ptr;
};


__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    glm::vec3 oc = r.orig - center;
    auto a = pow(glm::length(r.dir), 2);
    auto half_b = glm::dot(oc, r.dir);
    auto c = pow(glm::length(oc), 2) - radius*radius;

    auto discriminant = half_b*half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(root);
    glm::vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    get_sphere_uv(rec.p, rec.u, rec.v);
    return true;
}

__device__ bool sphere::bounding_box(float time0, float time1, AxisAllignedBoundingBox& outbox)const {
    outbox = AxisAllignedBoundingBox(center - glm::vec3(radius),center+glm::vec3(radius));
    return true;
}