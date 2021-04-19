#pragma once

#include "rtweekend.h"
#include "hittable.h"

#define RANDVEC3 glm::vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ glm::vec3 random_in_unit_sphere(curandState* local_rand_state) {
    
    float alpha = pi * (curand_uniform(local_rand_state) - 0.5f);
    float theta = 2 * pi * curand_uniform(local_rand_state);
    glm::vec3 p(cos(theta) * sin(alpha), sin(theta) * sin(alpha), cos(alpha));
    return p;
}

__device__ glm::vec3 random_unit_vector(curandState* local_rand_state)
{
    return glm::normalize(glm::vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)));
}


struct hit_record;


class material {
public:
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, curandState* local_rand_state
    ) const = 0;
};

class lambertian : public material {

public:
    __device__ lambertian(const glm::vec3& a) : albedo(a) {}


    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, curandState* local_rand_state)const override {

        glm::vec3 scatterDir = rec.normal + random_unit_vector(local_rand_state);
        if (nearZero(scatterDir))
            scatterDir = rec.normal;
        scattered = ray(rec.p, scatterDir);
        attenuation = albedo;
        return true;
    }
    
public:
    glm::vec3 albedo;
};

class metal : public material {

public:
    __device__ metal(const glm::vec3& a , float f) : albedo(a), fuzz(f) {}

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, curandState* local_rand_state)const override {
        glm::vec3 reflected = glm::normalize(glm::reflect(r_in.dir, rec.normal));
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return glm::dot(scattered.dir, rec.normal) > 0;
    }

public:
    glm::vec3 albedo;
    float fuzz;
};