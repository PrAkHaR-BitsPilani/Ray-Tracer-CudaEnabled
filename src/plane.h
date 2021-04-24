#pragma once

#include "rtweekend.h"
#include "hittable.h"


class plane : public hittable {

public:

    __device__ plane()
        : p(glm::vec3(0.0f)), x_axis(glm::vec3(1,0,0)), y_axis(glm::vec3(0,1,0)), x_range(1), y_range(1), mat_ptr(nullptr) {
        normal = glm::cross(x_axis, y_axis);
    }

    __device__ plane(glm::vec3 point, glm::vec3 xAxis, glm::vec3 yAxis, float xRange, float yRange, material* m)
        : p(point), x_range(xRange), y_range(yRange), mat_ptr(m) {
        x_axis = glm::normalize(xAxis);
        y_axis = glm::normalize(yAxis);
        normal = glm::cross(x_axis, y_axis);
    }

    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, AxisAllignedBoundingBox& outbox) const override;

public:
    glm::vec3 p;
    glm::vec3 x_axis;
    glm::vec3 y_axis;
    glm::vec3 normal;
    float x_range;
    float y_range;
    material* mat_ptr;
};


__device__ bool plane::hit(const ray& r, float t_min, float t_max, hit_record& rec) const 
{
    auto denom = glm::dot(r.dir, normal);
    if (denom == 0.0f)
        return false;
    auto t = glm::dot(normal, p - r.orig) / denom;
    if (t < t_min || t > t_max)
        return false;

    auto intersectionPoint = r.at(t);
    auto distX = glm::dot((intersectionPoint - p), x_axis);
    auto distY = glm::dot((intersectionPoint - p), y_axis);

    if (fabs(distX) > fabs(x_range) || fabs(distY) > fabs(y_range))
        return false;

    rec.t = t;
    rec.p = intersectionPoint;
    rec.mat_ptr = mat_ptr;
    rec.set_face_normal(r, normal);
    return true;
}

__device__ bool plane::bounding_box(float time0, float time1, AxisAllignedBoundingBox& outbox)const {
    return false;
}