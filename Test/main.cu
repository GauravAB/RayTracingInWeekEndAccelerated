#include <stdlib.h>
#include "cuda_runtime.h"
#include <iostream>
#include <time.h>
#include "device_launch_parameters.h"
#include <fstream>
#include "vector.h"
#include "ray.h"

#define checkCudaErrors(val) check_cuda((val),#val, __FILE__ , __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(1);
	}
}

__device__ vec3 color(const ray& r)
{
	vec3 unit_direction = unit_vector(r.direction());
	float  t = 0.5f * (unit_direction.y() + 1.0f);
	

	return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3* fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x + i ;
	
	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);

	ray r(origin, lower_left_corner + u * horizontal + v * vertical);

	fb[pixel_index] = color(r);
}


int main()
{
	std::ofstream out("image.ppm");

	int nx = 1200;
	int ny = 600;
	int tx = 8;
	int ty = 8;
	
	std::cerr << "Redering a " << nx << "x" << ny << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n"; 

	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);

	//allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void**)& fb, fb_size));
	clock_t start, stop;
	start = clock();

	//render on the buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	render << <blocks, threads >> > (fb, nx, ny, vec3(-2.0,-1.0,-1.0),vec3(4.0,0.0,0.0),vec3(0.0,2.0,0.0),vec3(0.0,0.0,0.0));

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();

	//time eval
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;

	std::cerr << "took" << timer_seconds << " seconds.\n";

	//image output
	out << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; j--)
	{
		for (int i = 0; i < nx; i++)
		{
			size_t pixel_index = j * nx + i;
			
			int ir = int(255.99 * fb[pixel_index].r());
			int ig = int(255.99 * fb[pixel_index].g());
			int ib = int(255.99 * fb[pixel_index].b());
			
			out << ir << " " << ig << " " << ib << std::endl;
		}
	}

	checkCudaErrors(cudaFree(fb));
	out.close();
}
















	