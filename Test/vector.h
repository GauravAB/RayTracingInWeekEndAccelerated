#pragma once
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "device_launch_parameters.h"

//creating common class for cpu and gpu

class vec3
{
public:
	float e[3];
	__host__ __device__ vec3() {}
	__host__ __device__ vec3(float e1, float e2, float e3)
	{
		e[0] = e1;
		e[1] = e2;
		e[2] = e3;
	}
	__host__ __device__ float x() { return e[0]; }
	__host__ __device__ float y() { return e[1]; }
	__host__ __device__ float z() { return e[2]; }
	__host__ __device__ float r() { return e[0]; }
	__host__ __device__ float g() { return e[1]; }
	__host__ __device__ float b() { return e[2]; }
	
	__host__ __device__ float operator[] (int index) { return e[index]; }
	__host__ __device__ const vec3& operator+() { return *this; }
	__host__ __device__ const vec3 operator-() { return vec3(-e[0], -e[1], -e[2]); }
	
	__host__ __device__ inline vec3& operator+=(const vec3& v);
	__host__ __device__ inline vec3& operator-=(const vec3& v);
	__host__ __device__ inline vec3& operator*=(const vec3& v);
	__host__ __device__ inline vec3& operator/=(const vec3& v);
	__host__ __device__ inline vec3& operator*=(const float f);
	__host__ __device__ inline vec3& operator/=(const float f);

	__host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
	__host__ __device__ inline float squared_length() const
	{
		return (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	}
	__host__ __device__ inline void make_unit_vector();
};


__host__ __device__ inline vec3& vec3::operator+=(const vec3& v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];

	return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v)
{
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];

	return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3& v)
{
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];

	return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3& v)
{
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];

	return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float f)
{
	e[0] *= f;
	e[1] *= f;
	e[2] *= f;

	return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float f)
{
	float k = 1.0f / (float)f;

	e[0] *= k;
	e[1] *= k;
	e[2] *= k;

	return *this;
}

__host__ __device__ void vec3::make_unit_vector()
{
	float k = 1.0f / float(sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]));

	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
}

inline std::istream& operator>>(std::istream& is, vec3& t)
{
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

inline std::ostream& operator<<(std::ostream& os, vec3& t)
{
	os << t.e[0] << " " << t.e[1] << " " << t.e[2];
	return os;
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2)
{
	return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const float t, const vec3& v)
{
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t)
{
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t)
{
	return vec3(v.e[0]/t, v.e[1]/t,v.e[2]/t);
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2)
{
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2)
{
	return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
		(-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
		(v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
	return v / v.length();
}






















