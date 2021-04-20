#pragma once
#include <cmath>
#include <cstdlib>
#include <limits>

// Common Headers

#include "ray.h"
#include "glm/glm.hpp"

#include <curand_kernel.h>


// Using
using std::sqrt;

// Constants

__managed__ float infinity = std::numeric_limits<float>::infinity();
__managed__ float pi = 3.1415926535897;

// Utility Functions

inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

__device__ inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

inline float random_float() {
    // Returns a random real in [0,1).
    return (float)(rand() / (RAND_MAX + 1.0));
}

__host__ __device__ inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

 inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return (int)(random_float(min, max + 1));
}

inline glm::vec3 random_in_unit_sphere()
{
    float alpha = random_float(-pi/2 , pi/2);
    float theta = random_float(0 , 2*pi);
    return glm::vec3(cos(theta) * sin(alpha), sin(theta) * sin(alpha), cos(alpha));

}

inline glm::vec3 randomColor()
{
    return glm::vec3(random_float(0, 1), random_float(0, 1), random_float(0, 1));
}

inline glm::vec3 random_unit_vector()
{
    return glm::normalize(random_in_unit_sphere());
}

__device__ bool nearZero(glm::vec3 e)
{
    const auto s = 1e-8;
    return (fabs(e.x) < s) && (fabs(e.y) < s) && (fabs(e.z) < s);
}
