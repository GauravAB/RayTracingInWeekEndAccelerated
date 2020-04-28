#pragma once
#include "ray.h"
#define M_PI 3.14159265

class camera
{
public:
	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;

	__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect) 
	{
		vec3 u, v, w;
		float theta = vfov * M_PI / 180;
		float half_height = tan(theta / 2);
		float half_width = half_height * aspect;
		origin = lookfrom;
		//camera basis orthonormal axis
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup,w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * u - half_height * v - w;
		horizontal = 2 * half_width * u;
		vertical = 2 * half_height * v;
	}
	
	__device__ ray get_ray(float s , float t)
	{
		return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
	}
};


