

/*
C++ HEADERS INCLUDE
*/
#include <iostream>


/*
CUDA INCLUDE
*/
#include "device_launch_parameters.h"
#include <driver_types.h>
#include <cuda_runtime.h>
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
#include "BVH_Node.h"
#include "rect.h"
#include "box.h"
#include "flip_face.h"
#include "transform.h"

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

__device__ glm::vec3 color_ray(const ray& r, hittable_list** world, int depth, curandState* local_rand_state)
{
	ray cur_ray = r;
	glm::vec3 cur_attenuation(1.0f);
	glm::vec3 cur_emitted(0.0);
	glm::vec3 day_time(1);
	glm::vec3 sky;
	while (depth--)
	{

		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
		{

			/*if (hitLightSource(cur_ray,rec.t))
				return cur_attenuation;*/

			ray scattered;
			glm::vec3 attenuation;
			glm::vec3 emitted = rec.mat_ptr->emitted(cur_ray, rec, rec.u, rec.v, rec.p);
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
			{
				cur_attenuation *= attenuation;
				cur_emitted += emitted * cur_attenuation;
				cur_ray = scattered;
			}
			else return cur_emitted + emitted * cur_attenuation;
		}
		else {

			glm::vec3 dir = glm::normalize(cur_ray.dir);
			float t = 0.5f * (dir.y + 1.0f);
			//sky = ((1.0f - t) * glm::vec3(0.8, 0.2, 0.9) + t * glm::vec3(0.1, 0.2, 0.9));
			sky = ((1.0f - t) * glm::vec3(1) + t * glm::vec3(0.0));
			
			return cur_emitted;// +cur_attenuation * day_time * sky;
			
		}
	}

	return cur_emitted;
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

__global__ void render(unsigned char* fb, int max_x, int max_y, camera cam, hittable_list** world, curandState* rand_state, int loopIndex, int loopCount, int number_of_samples, int depth)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x + loopIndex * blockDim.x * loopCount;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
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
	fb[pixel_index + 2] = 255 * clamp(sqrt(pixelColor.b / number_of_samples), 0, 0.999f);
	fb[pixel_index + 0] = 255 * clamp(sqrt(pixelColor.r / number_of_samples), 0, 0.999f);
	fb[pixel_index + 1] = 255 * clamp(sqrt(pixelColor.g / number_of_samples), 0, 0.999f);
}

#define RND (curand_uniform(&local_rand_state))	

__global__ void createCornellBox(hittable** d_list, hittable_list** d_world)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		int i = 0;
		auto red = new lambertian(glm::vec3(0.65, 0.05, 0.05));
		auto green = new lambertian(glm::vec3(0.12, 0.45, 0.15));
		auto light = new diffuse_light(glm::vec3(15));
		auto white = new lambertian(glm::vec3(0.73));
		auto magenta = new lambertian(glm::vec3(0.8, 0.2, 0.9));
		auto blue = new metal(glm::vec3(0.1, 0.2, 0.9), 0.5);

		d_list[i++] = new yz_rect(0, 10, 0, 10, 10, green);
		d_list[i++] = new yz_rect(0, 10, 0, 10, 0, red);
		d_list[i++] = new flip_face(new xz_rect(3, 7, 3, 7, 9.5, light));
		/*d_list[i++] = new xz_rect(10, 110, 445, 545, 545, light);
		d_list[i++] = new xz_rect(445, 545, 10, 110, 554, light);
		d_list[i++] = new xz_rect(445, 545, 445, 545, 554, light);*/
		d_list[i++] = new xz_rect(0, 10, 0, 10, 0, white);
		d_list[i++] = new xz_rect(0, 10, 0, 10, 10, white);
		d_list[i++] = new xy_rect(0, 10, 0, 10, 10, white);

		//d_list[i++] = new sphere(glm::vec3(50, 10, 30), 10, white);
		
		d_list[i++] = new RotateY(new box(glm::vec3(130, 0, 65) * 0.018f, glm::vec3(295, 165, 230) * 0.018f, blue),0);
		d_list[i++] = new box(glm::vec3(265, 0, 295) * 0.018f, glm::vec3(430, 330, 460) * 0.018f, blue);


		*d_world = new hittable_list(d_list, 8);
	}

}

__global__ void create_world(hittable** d_list, hittable_list** d_world, int limit, curandState* rand_state) {
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
					d_list[i++] = new sphere(centre, 0.2, new alienMat(glm::vec3(0.8,0.2,0.9),0.0f));
					//d_list[i++] = new sphere(centre, 0.2, new metal(glm::vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
					//d_list[i++] = new sphere(centre, 0.2, new lambertian(glm::vec3(RND * RND, RND * RND, RND * RND)));
					//d_list[i++] = new sphere(centre, 0.2, new lambertian(glm::vec3(RND, RND, RND)));
				}
				else {
					d_list[i++] = new sphere(centre, 0.2, new alienMat(glm::vec3(0.1,0.2,0.9),0.0f));
					//d_list[i++] = new sphere(centre, 0.2, new metal(glm::vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
				}
			}
		}

		d_list[i++] = new sphere(glm::vec3(0, -1000, 0), 1000, new metal(glm::vec3(0.1f),0.2f));

		d_list[i++] = new sphere(glm::vec3(-4, 1, 0), 0, new metal(glm::vec3(0.2, 0.9, 0.1),0.0f));
		d_list[i++] = new sphere(glm::vec3(0, 1, 0), 0, new metal(glm::vec3(0.2, 0.9, 0.1),0.0f));
		d_list[i++] = new sphere(glm::vec3(4, 1, 0), 0, new metal(glm::vec3(0.2, 0.9, 0.1), 0.0f));
		d_list[i++] = new xy_rect(-3, 3, 0, 4, -7, new diffuse_light(glm::vec3(4,0,4)));

		*rand_state = local_rand_state;


		*d_world = new hittable_list(d_list, 4 * limit * limit + 1 + 4);
	}
}

void DisplayHeader()
{
	using namespace std;

	const int kb = 1024;
	const int mb = kb * kb;
	cout << "RayTracing.GPU" << endl << "=========" << endl << endl;

	cout << "CUDA version:   v" << CUDART_VERSION << endl;
	//cout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl;

	int devCount;
	cudaGetDeviceCount(&devCount);
	cout << "CUDA Devices: " << endl << endl;

	for (int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		cout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
		cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
		cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
		cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
		cout << "  Block registers: " << props.regsPerBlock << endl << endl;

		cout << "  Warp size:         " << props.warpSize << endl;
		cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
		cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
		cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
		cout << endl;
	}
}

int main()
{

	DisplayHeader();

	stbi_flip_vertically_on_write(1);

	const float aspectRatio = 1;// 3840.0f / 2160.0f;
	const int imageWidth = 600;
	const int imageHeight = imageWidth / aspectRatio;
	int num_of_pixels = imageWidth * imageHeight;
	int number_of_samples = 100;
	int depth = 10;
	int limit = 3;
	int tx = 16;
	int ty = 16;

	//CAMERA
	camera cam(glm::vec3(5, 5, -10), glm::vec3(5, 5, 0), glm::vec3(0, 1, 0), 50, aspectRatio);

	std::cerr << "Rendering a " << imageWidth << "x" << imageHeight << " image \n";
	std::cerr << "Number of blocks: " << (imageWidth / tx + 1) * (imageHeight / ty + 1) << "\n";
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

	rand_init << <1, 1 >> > (d_rand_state2);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());

	//WORLD
	hittable** d_list;
	CudaCall(cudaMallocManaged(&d_list, (8) * sizeof(hittable*)));
	hittable_list** d_world;
	CudaCall(cudaMallocManaged(&d_world, sizeof(hittable_list*)));
	createCornellBox << <1, 1 >> > (d_list, d_world);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());

	/*hittable_list* d_world_copy;
	CudaCall(cudaMallocHost(&d_world_copy, sizeof(hittable_list)));
	CudaCall(cudaMemcpy((void*)d_world_copy, (void*)*d_world, sizeof(hittable_list*), cudaMemcpyDeviceToHost));*/

	BVH_Node* root;
	CudaCall(cudaMallocHost((void**)&root, sizeof(BVH_Node)));
	//root = new BVH_Node(*d_world_copy, 0, 1);

	//TIMING
	clock_t start, stop;
	start = clock();

	int loopCount = 1;

	dim3 blocks(((imageWidth + tx - 1) / tx + loopCount - 1) / loopCount, (imageHeight + ty - 1) / ty);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (imageWidth, imageHeight, d_rand_state);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());
	/*render << <blocks, threads >> > (fb, imageWidth, imageHeight, cam, d_world, d_rand_state, 0, block_per_loop, number_of_samples, depth);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());*/
	for (int i = 0; i < loopCount; i++)
	{
		render << <blocks, threads >> > (fb, imageWidth, imageHeight, cam, d_world, d_rand_state, i, ((imageWidth + tx - 1) / tx + loopCount - 1) / loopCount, number_of_samples, depth);
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
	//freeCornellBox << <1, 1 >> > (d_list, d_world);
	CudaCall(cudaGetLastError());
	CudaCall(cudaFree(d_list));
	CudaCall(cudaFree(d_world));
	CudaCall(cudaFree(fb));

	cudaDeviceReset();
}

