

/*
C++ HEADERS INCLUDE
*/
#include <iostream>


/*
CUDA INCLUDE
*/
#include "device_launch_parameters.h"
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>


/*
HEADER INCLUDE
*/
#include "ray.h"
#include "camera.h"
#include "material.h"
#include "sphere.h"
#include "plane.h"
#include "hittable_list.h"

/*
EXTERNAL APIs
*/
#include <stb_image/stb_image_write.h>


/*
CUDA DEBUGGING SNIPPET
CHECKS CUDA ERRORS
*/
#define CudaCall(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(1);
	}
}


__device__ bool hitLightSource(const ray& r, float t_max)
{
	glm::vec3 p(0,5,0);
	glm::vec3 x_axis(0,0,1);
	glm::vec3 y_axis(-1,0,0);
	glm::vec3 normal = glm::cross(x_axis , y_axis);
	float x_range = 2.0f;
	float y_range = 2.0f;
	float t_min = 0.001f;

	auto denom = glm::dot(r.dir, normal);
	if (denom == 0.0f)
		return false;
	auto t = glm::dot(normal, p - r.orig) / denom;
	if (t < t_min || t > t_max)
		return false;

	auto intersectionPoint = r.at(t);
	auto distX = glm::dot((intersectionPoint - p), x_axis);
	auto distY = glm::dot((intersectionPoint - p), y_axis);

	if (fabs(distX) > fabs(x_range) || fabs(distY) > fabs(y_range))
		return false;
	return true;
}

__device__ glm::vec3 color_ray(const ray& r, hittable** world, int depth, curandState* local_rand_state)
{
	ray cur_ray = r;
	glm::vec3 cur_attenuation(1.0f);
	while(depth--)
	{

		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
		{
			
			/*if (hitLightSource(cur_ray,rec.t))
				return cur_attenuation;*/

			ray scattered;
			glm::vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
			{
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else return glm::vec3(0);
		}
		else{

			/*if (hitLightSource(cur_ray, FLT_MAX))
				return cur_attenuation;*/

			glm::vec3 dir = glm::normalize(r.dir);
			float t = 0.5f * (dir.y + 1.0f);
			return glm::vec3(1.0) * cur_attenuation * ((1.0f - t) * glm::vec3(1.0, 1.0, 1.0) + t * glm::vec3(0.5, 0.7, 1.0));
		}
	}

	return glm::vec3(0);
}

__global__ void rand_init(curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = 1 * (j * max_x + i);
	curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(unsigned char* fb, int max_x, int max_y, camera cam, hittable** world, curandState* rand_state, int loopIndex, int loopCount, int number_of_samples, int depth)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + loopIndex * blockDim.x * loopCount;
	int j = threadIdx.y + blockIdx.y* blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = (j * max_x + i);
	curandState local_rand_state = rand_state[pixel_index];
	pixel_index *= 3;
	glm::vec3 pixelColor(0.0f);
	for (int s = 0; s < number_of_samples; s++)
	{
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = cam.get_ray(u, v);
		pixelColor += color_ray(r, world, depth, &local_rand_state);
	}
	fb[pixel_index + 2] = 255 * clamp(sqrt(pixelColor.b/number_of_samples),0,0.999f);
	fb[pixel_index + 0] = 255 * clamp(sqrt(pixelColor.r/number_of_samples),0,0.999f);
	fb[pixel_index + 1] = 255 * clamp(sqrt(pixelColor.g/number_of_samples),0,0.999f);
}

#define RND (curand_uniform(&local_rand_state))	

__global__ void create_world(hittable** d_list, hittable** d_world, int limit, curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) 
	{

		curandState local_rand_state = *rand_state;

		int i = 0;

		for (int a = -limit; a < limit; a++)
		{
			for (int b = -limit; b < limit; b++)
			{
				float material_deciding_factor = RND;
				glm::vec3 centre(a + RND, 0.2, b + RND);
				if (material_deciding_factor >= 0.8f)
				{
					d_list[i++] = new sphere(centre, 0.2, new lambertian(glm::vec3(RND * RND, RND * RND, RND * RND)));
					//d_list[i++] = new sphere(centre, 0.2, new lambertian(glm::vec3(RND, RND, RND)));
				}
				else {
					d_list[i++] = new sphere(centre , 0.2, new metal(glm::vec3(0.5f*(1.0f+RND),0.5f*(1.0f+RND),0.5f*(1.0f+RND)), 0.5f * RND));
				}
			}
		}

		d_list[i++] = new sphere(glm::vec3(0, -1000, 0), 1000, new lambertian(glm::vec3(0.5f)));

		d_list[i++] = new sphere(glm::vec3( 0, 1, 0), 1, new metal(glm::vec3(0.7, 0.5, 0.4),0.0f));
		d_list[i++] = new sphere(glm::vec3(-4, 1, 0), 1, new metal(glm::vec3(0.7, 0.5, 0.4),0.0f));
		d_list[i++] = new sphere(glm::vec3( 4, 1, 0), 1, new metal(glm::vec3(0.7, 0.5, 0.4),0.0f));
		d_list[i++] = new plane(glm::vec3(0,0,-5),glm::vec3(1,0,0),glm::vec3(0,1,0),20,20, new metal(glm::vec3(0.7,0.5,0.4), 0.0));

		*rand_state = local_rand_state;
		*d_world = new hittable_list(d_list, 4 * limit * limit + 1 + 4);
	}
}

__global__ void free_world(hittable** d_list, hittable** d_world, int limit) {
	int i = 0;
	for (; i < 4 * limit * limit; i++)
	{
		delete ((sphere*)d_list[i])->mat_ptr;
		delete d_list[i];
	}

	delete ((sphere*)d_list[i])->mat_ptr;
	delete d_list[i++];

	delete ((sphere*)d_list[i])->mat_ptr;
	delete d_list[i++];

	delete ((sphere*)d_list[i])->mat_ptr;
	delete d_list[i++];

	delete ((sphere*)d_list[i])->mat_ptr;
	delete d_list[i++];

	delete ((plane*)d_list[i])->mat_ptr;
	delete d_list[i++];

	delete* d_world;
}


int main()
{
	stbi_flip_vertically_on_write(1);

	const float aspectRatio = 3840.0f / 2160.0f;
	const int imageWidth = 3840;
	const int imageHeight = imageWidth / aspectRatio;
	int num_of_pixels = imageWidth * imageHeight;
	int number_of_samples = 10;
	int depth = 10;
	int limit = 3;
	int tx = 16;
	int ty = 16;

	//CAMERA
	camera cam(glm::vec3(-3, 2, 6), glm::vec3(0,0,-5), glm::vec3(0, 1, 0), 45, aspectRatio);

	std::cerr << "Rendering a " << imageWidth << "x" << imageHeight << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";
	std::cerr << "Number of blocks: " << (imageWidth / tx + 1) * ( imageHeight / ty + 1) << "\n";
	std::cerr << "Number of threads per block: " << tx * ty << "\n";
	std::cerr << "Samples per pixel: " << number_of_samples << "\n";
	std::cerr << "Max depth of ray bounce: " << depth << "\n";

	size_t fb_size = 3 * num_of_pixels * sizeof(unsigned char);
	unsigned char* fb;
	CudaCall(cudaMallocManaged((void**)&fb, fb_size));

	curandState* d_rand_state;
	CudaCall(cudaMalloc((void**)&d_rand_state, num_of_pixels * sizeof(curandState)));
	curandState* d_rand_state2;
	CudaCall(cudaMalloc((void**)&d_rand_state2, num_of_pixels * sizeof(curandState)));

	rand_init <<<1, 1 >>> (d_rand_state2);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());

	//WORLD
	hittable** d_list;
	CudaCall(cudaMallocManaged((void**)&d_list, (4 * limit * limit + 1 + 4 )*sizeof(hittable*)));
	hittable** d_world;
	CudaCall(cudaMallocManaged((void**)&d_world, sizeof(hittable*)));
	create_world <<<1,1>>> (d_list, d_world,limit,d_rand_state2);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());

	//TIMING
	clock_t start, stop;
	start = clock();

	int loopCount = 1;

	dim3 blocks(((imageWidth+tx-1)/tx + loopCount - 1)/loopCount, (imageHeight+ty-1)/ty);
	dim3 threads(tx, ty);
	render_init <<<blocks, threads>>> (imageWidth, imageHeight, d_rand_state);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());
	/*render << <blocks, threads >> > (fb, imageWidth, imageHeight, cam, d_world, d_rand_state, 0, block_per_loop, number_of_samples, depth);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());*/
	for (int i = 0; i < loopCount; i++)
	{
		render <<<blocks, threads >>> (fb, imageWidth, imageHeight, cam, d_world, d_rand_state, i, ((imageWidth + tx - 1) / tx + loopCount - 1) / loopCount, number_of_samples, depth);
		CudaCall(cudaGetLastError());
		CudaCall(cudaDeviceSynchronize());
	}

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "Took " << timer_seconds << " seconds.\n";

	int result = stbi_write_png("res/images/output.png", imageWidth, imageHeight, 3, fb, imageWidth * 3);
	std::cout << "PNG Created : " << result << "\n";

	// clean up
	CudaCall(cudaDeviceSynchronize());
	free_world <<<1, 1>>> (d_list, d_world,limit);
	CudaCall(cudaGetLastError());
	CudaCall(cudaFree(d_list));
	CudaCall(cudaFree(d_world));
	CudaCall(cudaFree(fb));

	cudaDeviceReset();


}