#pragma once
#include "device_launch_parameters.h"
#include "vector.h"

class ray
{

public:
	vec3 A;
	vec3 B;

	__device__ ray() {}
	__device__ ray(const vec3& a, const vec3& b) { A = a; B = b; }
	__device__ vec3 origin() const { return A; }
	__device__ vec3 direction() const { return B; }
	__device__ vec3 point_at_parameter(float t) { return A + t * B; }
	
};