#pragma once

#include "rtweekend.h"

class AxisAllignedBoundingBox {
	
public:
	 __device__ AxisAllignedBoundingBox()
		: minimum(glm::vec3(0)) , maximum(glm::vec3(0)) {}
	 __device__ AxisAllignedBoundingBox(const glm::vec3& mini, const glm::vec3& maxi)
		:minimum(mini), maximum(maxi) {}
	
	__device__ bool hit(const ray& r, float t_min, float t_max) const
	{
		for (int a = 0; a < 3; a++) {
			auto invD = 1.0f / r.dir[a];
			auto t0 = (minimum[a] - r.orig[a]) * invD;
			auto t1 = (maximum[a] - r.orig[a]) * invD;
			if (invD < 0.0f)
			{
				auto temp = t0;
				t0 = t1;
				t1 = temp;
			}
			t_min = t0 > t_min ? t0 : t_min;
			t_max = t1 < t_max ? t1 : t_max;
			if (t_max <= t_min)
				return false;
		}
		return true;

	}

public:
		glm::vec3 minimum;
		glm::vec3 maximum;
};

__device__ AxisAllignedBoundingBox surroundingBox(AxisAllignedBoundingBox& box0, AxisAllignedBoundingBox& box1)
{
	glm::vec3 start(fmin(box0.minimum[0], box1.minimum[0]), fmin(box0.minimum[1], box1.minimum[1]), fmin(box0.minimum[2], box1.minimum[2]));

	glm::vec3 end(fmax(box0.maximum[0], box1.maximum[0]), fmax(box0.maximum[1], box1.maximum[1]), fmax(box0.maximum[2], box1.maximum[2]));

	return AxisAllignedBoundingBox(start, end);

}