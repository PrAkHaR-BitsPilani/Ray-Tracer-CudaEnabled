#pragma once

#include "rtweekend.h"

#include "hittable.h"

#include <memory>
#include <vector>


class hittable_list : public hittable {
public:
    __host__ __device__ hittable_list() {}
    __device__ hittable_list(hittable** l, int n) { list = l; list_size = n; }
    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;

private:
    hittable** list;
    int list_size;
};


__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;    
    auto hit_anything = false;
    auto closest_so_far = t_max;

    for(int i = 0 ; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}