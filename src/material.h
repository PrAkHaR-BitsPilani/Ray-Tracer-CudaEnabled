#pragma once

#include "rtweekend.h"
#include "hittable.h"
#include "texture.h"

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

    __device__ virtual glm::vec3 emitted(const ray& in , const hit_record& rec, float u, float v, const glm::vec3& p)const
    {
        return glm::vec3(0);
    }

};

class lambertian : public material {

public:
    __device__ lambertian(const glm::vec3& a) : albedo(new solid_color(a)) {}

    //__device__ lambertian(texture* a) : albedo(a) {}


    __device__ bool scatter(
        const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, curandState* local_rand_state)const override {

        glm::vec3 scatterDir = rec.normal + random_in_unit_sphere(local_rand_state);
        if (nearZero(scatterDir))
            scatterDir = rec.normal;
        scattered = ray(rec.p, scatterDir);
        attenuation = albedo->value(rec.u , rec.v , rec.p);
        return true;
    }
    
public:
    Texture* albedo;
};

class metal : public material {

public:
    __device__ metal(const glm::vec3& a , float f) : albedo(a), fuzz(f) {}

    __device__ bool scatter(
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

class diffuse_light : public material {

public:
    __device__ diffuse_light(Texture* t) : emit(t) {}
    __device__ diffuse_light(glm::vec3 color) : emit(new solid_color(color)) {}

    __device__ bool scatter(
        const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, curandState* local_rand_state
    ) const override {
        return false;
    }

    __device__ glm::vec3 emitted(const ray& in, const hit_record& rec, float u, float v, const glm::vec3& p)const override {
        if (rec.front_face)
            return emit->value(u, v, p);
        return glm::vec3(0);
    }




public:
    Texture* emit;
};

class alienMat : public material {
public:

    __device__ alienMat(const glm::vec3& a, float f ) : albedo(a), fuzz(f), emit(new solid_color(a)) {}

    __device__ bool scatter(
        const ray& r_in, const hit_record& rec, glm::vec3& attenuation, ray& scattered, curandState* local_rand_state)const override {
        glm::vec3 reflected = glm::normalize(glm::reflect(r_in.dir, rec.normal));
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return glm::dot(scattered.dir, rec.normal) > 0;
    }

    __device__ glm::vec3 emitted(const ray& in , const hit_record& rec, float u, float v, const glm::vec3& p)const override {
        return emit->value(u, v, p);
    }

public:
    glm::vec3 albedo;
    float fuzz;
    Texture* emit;
};