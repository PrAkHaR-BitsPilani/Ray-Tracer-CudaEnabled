#pragma once
#include <cmath>
#include <cstdlib>
#include <limits>

// Common Headers

#include <curand_kernel.h>


// Using
using std::sqrt;

// Constants

__managed__ float infinity = std::numeric_limits<float>::infinity();
__managed__ float pi = 3.1415926535897;

// Utility Functions

__device__ float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

__device__ inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__ inline int random_int(int a , int b , curandState& local_rand_state) {
    return a + int(b * curand_uniform(&local_rand_state));
}

__device__ inline float random_float(curandState& local_rand_state) {
    // Returns a random real in [0,1).
    return curand_uniform(&local_rand_state);
}

__device__ inline float random_float(float min, float max, curandState& local_rand_state) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float(local_rand_state);
}

