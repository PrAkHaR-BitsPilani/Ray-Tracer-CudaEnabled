#pragma once

#include "rtweekend.h"

class AxisAllignedBoundingBox {
	
public:
	__device__ AxisAllignedBoundingBox()
		: minimum(glm::vec3(0)) , maximum(glm::vec3(0)) {}
	__device__ AxisAllignedBoundingBox(const glm::vec3& a, const glm::vec3& b)
		:minimum(a), maximum(b) {}
	

	__device__ bool hit(const ray& r, float t_min, float t_max)
	{

	}

public:
		glm::vec3 minimum;
		glm::vec3 maximum;
};
