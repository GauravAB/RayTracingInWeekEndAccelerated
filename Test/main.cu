#include <stdlib.h>
#include "cuda_runtime.h"
#include <iostream>
#include <time.h>
#include "device_launch_parameters.h"
#include <fstream>
#include "vector.h"

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

__global__ void render(float* fb, int max_x, int max_y)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x * 3 + i * 3;
	fb[pixel_index + 0] = float(i) / max_x;
	fb[pixel_index + 1] = float(j) / max_y;
	fb[pixel_index + 2] = 0.2;
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
	size_t fb_size = 3 * num_pixels * sizeof(float); // RGB * row*col* float size

	//allocate FB
	float* fb;
	checkCudaErrors(cudaMallocManaged((void**)& fb, fb_size));
	clock_t start, stop;
	start = clock();

	//render on the buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	render << <blocks, threads >> > (fb, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();

	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;

	std::cerr << "took" << timer_seconds << " seconds.\n";

	//image output
	out << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; j--)
	{
		for (int i = 0; i < nx; i++)
		{
			size_t pixel_index = j * 3 * nx + i * 3;
			vec3 col(fb[pixel_index + 0], fb[pixel_index + 1], fb[pixel_index + 2]);
			
			int ir = int(255.99 * col[0]);
			int ig = int(255.99 * col[1]);
			int ib = int(255.99 * col[2]);
			
			out << ir << " " << ig << " " << ib << std::endl;
		}
	}

	checkCudaErrors(cudaFree(fb));
	out.close();
}
















	