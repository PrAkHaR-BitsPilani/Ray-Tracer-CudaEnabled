#pragma once


#include "vec3.cuh"
#include <crt/host_defines.h>


class ray {
public:

    __device__ ray() : orig(glm::vec3(0.0f)) , dir(glm::vec3(0.0f)) , tm(0.0f) {}

    __device__ ray(const glm::vec3& origin, const glm::vec3& direction)
        : orig(origin), dir(direction), tm(0)
    {}


    __device__ ray(const glm::vec3& origin, const glm::vec3& direction, float time)
        : orig(origin), dir(direction), tm(time)
    {}

    __device__ glm::vec3 at(float t) const {
        return orig + t * dir;
    }

public:
    glm::vec3 orig;
    glm::vec3 dir;
    float tm;
};