#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <float.h>
#include <fstream>

#include <curand_kernel.h>
#include "vector.h"
#include "ray.h"
#include "hitable_list.h"
#include "sphere.h"
#include "hitable.h"
#include "camera.h"
#include "material.h"




#define checkCudaErrors(val) check_cuda((val),#val, __FILE__ , __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{

		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state)
{
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);

	for (int i = 0; i < 50; i++)
	{
		hit_record rec;

		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
		{
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
			{
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else
			{
				return vec3(0.0, 0.0, 0.0);
			}
		}
		else
		{
			vec3 unit_direction = unit_vector(r.direction());
			float  t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c =  (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			
			return cur_attenuation * c;
		}

	}
	
	return vec3(0.0, 0.0, 0.0);
}
__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y))
	{
		return;
	}
	else
	{
		int pixel_index = j * max_x + i;
		unsigned int seed = 1234;

		curand_init(seed, pixel_index, 0, &rand_state[pixel_index]);

	}

}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera **cam, hitable** world, curandState *rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i >= max_x) || (j >= max_y)) return;

	int pixel_index = j * max_x + i ;

	curandState local_rand_state = rand_state[pixel_index];

	vec3 col(0, 0, 0);
	
	for (int s = 0; s < ns; s++)
	{
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		ray r = (*cam)->get_ray(u, v);

		col += color(r, world, &local_rand_state);
	}

	rand_state[pixel_index] = local_rand_state;
	col /= float(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);

	
	fb[pixel_index] = col;
}

__global__ void create_world(hitable** d_list, hitable** d_world,camera** d_camera, int nx, int ny)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{	
		d_list[0] = new sphere(vec3(0, 0, -1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
		d_list[1] = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
		d_list[2] = new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.0));
		d_list[3] = new sphere(vec3(-1, 0, -1), 0.5, new dielectric(1.5));
		d_list[4] = new sphere(vec3(-1, 0, -1), -0.45, new dielectric(1.5));

		*d_world = new hitable_list(d_list, 5);
		*d_camera = new camera(vec3(-2,2,1),vec3(0,0,-1),vec3(0,1,0),20.0,float(nx)/float(ny));
	}
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera)
{
	for (int i = 0; i < 5; i++)
	{
		delete ((sphere*)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	
	delete* d_world;
	delete* d_camera;
}

int main()
{
	std::ofstream out("image.ppm");

	int nx = 1200;
	int ny = 600;
	int ns = 100;
	int tx = 8;
	int ty = 8;
	
	std::cerr << "Redering a " << nx << "x" << ny << " image with " << ns << "samples per pixel ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n"; 

	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);

	//allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void**)& fb, fb_size));
	
	//allocate random state
	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)& d_rand_state, num_pixels * sizeof(curandState)));
	

	//hitable world
	hitable** d_list;
	checkCudaErrors(cudaMalloc((void**)& d_list, 2 * sizeof(hitable*)));
	hitable** d_world;
	checkCudaErrors(cudaMalloc((void**)& d_world, sizeof(hitable*)));
	
	//device camera
	camera** d_camera;
	checkCudaErrors(cudaMalloc((void**)& d_camera, sizeof(camera*)));

	create_world << < 1, 1 >> > (d_list, d_world, d_camera,nx,ny);
	
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	
	
	clock_t start, stop;
	start = clock();

	//render on the buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	render_init <<< blocks, threads >>> (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	render <<<blocks, threads>>> (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();

	//time eval
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;

	std::cerr << "took: " << timer_seconds << " seconds.\n";
	
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
	

	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> > (d_list, d_world,d_camera);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_camera));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();

	out.close();

}
















	