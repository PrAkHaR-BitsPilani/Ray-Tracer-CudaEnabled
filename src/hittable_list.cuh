#pragma once

#include "rtweekend.cuh"
#include "hittable.cuh"

#include <memory>
#include <vector>


class hittable_list : public hittable {
public:
    __host__ __device__ hittable_list() {}
    __device__ hittable_list(hittable** l, int n) { list = l; list_size = n; allocated_size = n; }
    __device__ virtual bool hit(
        const ray& r, float t_min, float t_max, hit_record& rec) const override;
    __device__ virtual bool bounding_box(float time0, float time1, AxisAllignedBoundingBox& outbox) const override;
    __device__ void add(hittable* object);

public:
    hittable** list;
    int list_size;
    int allocated_size;
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

__device__ bool hittable_list::bounding_box(float time0, float time1, AxisAllignedBoundingBox& outbox)const {
    if (list_size == 0)
        return false;
    AxisAllignedBoundingBox tempBox;
    for (int i = 0; i < list_size; i++)
    {
        if (!list[i]->bounding_box(time0, time1, tempBox))return false;
        if (i)outbox = surroundingBox(outbox, tempBox);
        else outbox = tempBox;
    }
    return true;
}

__device__ void hittable_list::add(hittable* obj)
{
    if (list_size == 0)
    {
        list = new hittable * [1];
        list_size = allocated_size = 1;
        list[0] = obj;
    }
    else {
        if (allocated_size <= list_size) {
            hittable** new_list = new hittable * [list_size * 2];
            for (int i = 0; i < list_size; i++) {
                new_list[i] = list[i];
            }
            list = new_list;
            allocated_size = list_size * 2;
        }
        list[list_size++] = obj;
    }
}