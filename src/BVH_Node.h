#pragma once

#include "hittable_list.h"
#include "AxisAllignedBoundingBox.h"
#include <iostream>
#include <algorithm>

inline bool box_compare(hittable* a, hittable* b, int axis);
inline bool box_x_compare(hittable* a, hittable* b);
inline bool box_y_compare(hittable* a, hittable* b);
inline bool box_z_compare(hittable* a, hittable* b);

class BVH_Node : public hittable {
public:
	BVH_Node();

	BVH_Node(hittable_list list, float time_0, float time_1)
	: BVH_Node(list.list , 0 , list.list_size , time_0 , time_1){}

	BVH_Node(hittable** list, size_t start, size_t end, float time_0, float time_1) {
		auto objects = list;
		
		int axis = random_int(0, 2);
		auto comparator = (axis == 0) ? box_x_compare
			: (axis == 1) ? box_y_compare
			: box_z_compare;
		size_t object_span = end - start;
		
		if (object_span == 1)
		{
			left = right = objects[start];
		}
		else if (object_span == 2)
		{
			if (comparator(objects[start], objects[start + 1])){
				left = objects[start];
				right = objects[start + 1];
			}
			else {
				left = objects[start + 1];
				right = objects[start];
			}
		}
		else {
			std::sort(objects + start, objects + end, comparator);
			auto mid = start + object_span / 2;
			left = new BVH_Node(objects, start, mid, time_0, time_1);
			right = new BVH_Node(objects, mid + 1, end, time_0, time_1);
		}

		AxisAllignedBoundingBox aLeft, aRight;
		
		if (!left->bounding_box(time_0, time_1, aLeft) || !right->bounding_box(time_0, time_1, aRight))
			printf("No Bounding Box found in constructor!");
		box = surroundingBox(aLeft, aRight);

	}

	__device__ virtual bool hit(
		const ray& r, float t_min, float t_max, hit_record& rec) const override;

	virtual bool bounding_box(float time0, float time1, AxisAllignedBoundingBox& outbox) const override;

public:
	hittable* left;
	hittable* right;
	AxisAllignedBoundingBox box;

};

bool BVH_Node::bounding_box(float time0, float time1, AxisAllignedBoundingBox& outbox)const {
	outbox = box;
	return true;
}

__device__ bool BVH_Node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {

	if(box.hit(r,t_min,t_max) == false)
		return false;

	bool leftHit = left->hit(r, t_min, t_max,rec);
	bool rightHit = right->hit(r, t_min, leftHit ? rec.t : t_max,rec);
	return leftHit | rightHit;

}

inline bool box_compare(hittable* a, hittable* b, int axis) {
	AxisAllignedBoundingBox aa, bb;
	if (!a->bounding_box(0, 0, aa) || !b->bounding_box(0, 0, bb))
		printf("No bounding box found in constructor!");
	return aa.minimum[axis] < bb.minimum[axis];
}

bool box_x_compare(hittable* a, hittable* b)
{
	return box_compare(a, b, 0);
}

bool box_y_compare(hittable* a, hittable* b)
{
	return box_compare(a, b, 1);
}

bool box_z_compare(hittable* a, hittable* b)
{
	return box_compare(a, b, 2);
}