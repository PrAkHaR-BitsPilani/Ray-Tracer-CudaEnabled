

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
#include "ray.cuh"
#include "camera.cuh"
#include "material.cuh"
#include "sphere.cuh"
#include "plane.cuh"
#include "hittable_list.cuh"
#include "rect.cuh"
#include "box.cuh"
#include "flip_face.cuh"
#include "transform.cuh"

/*
EXTERNAL APIs
*/
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image.h>
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

//DEBUG VARIABLES
__managed__ bool defyPhysics = false;
__managed__ bool allowLightSource = true;

__device__ glm::vec3 color_ray(const ray& r, hittable** world, int depth, curandState* local_rand_state)
{
	ray cur_ray = r;
	glm::vec3 cur_attenuation(1.0f);
	glm::vec3 cur_emitted(0);
	glm::vec3 day_time(0.1);
	glm::vec3 sky;
	while (depth--)
	{

		hit_record rec;

		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
		{

			ray scattered;
			glm::vec3 attenuation;
			glm::vec3 emitted = rec.mat_ptr->emitted(cur_ray, rec, rec.u, rec.v, rec.p);
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
			{
				cur_attenuation *= attenuation;
				if (defyPhysics)
					cur_emitted += emitted;
				else
					cur_emitted += emitted * cur_attenuation;
				cur_ray = scattered;
			}
			else {
				if (allowLightSource)
					return cur_emitted + emitted * cur_attenuation;
				else return cur_attenuation;
			}
		}
		else {

			glm::vec3 dir = glm::normalize(cur_ray.dir);
			float t = 0.5f * (dir.y() + 1.0f);
			//sky = ((1.0f - t) * glm::vec3(0.8, 0.2, 0.9) + t * glm::vec3(0.1, 0.2, 0.9));
			sky = ((1.0f - t) * glm::vec3(1) + t * glm::vec3(0.5, 0.7, 1.0));
			if (defyPhysics)
				return (allowLightSource ? cur_emitted : glm::vec3(1)) * cur_attenuation * day_time * sky;
			else
				return (allowLightSource ? cur_emitted : glm::vec3(0)) + cur_attenuation * day_time * sky;
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
	curand_init(1984,pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(glm::vec3* fb, int max_x, int max_y, camera** cam, hittable** world, curandState* rand_state, int number_of_samples, int depth)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = (j * max_x + i);
	curandState local_rand_state = rand_state[pixel_index];
	glm::vec3 pixelColor(0.0f);
	for (int s = 0; s < number_of_samples; s++)
	{
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v);
		pixelColor += color_ray(r, world, depth, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	pixelColor /= float(number_of_samples);
	pixelColor[0] = clamp(sqrt(pixelColor[0]),0,0.999f);
	pixelColor[1] = clamp(sqrt(pixelColor[1]),0,0.999f);
	pixelColor[2] = clamp(sqrt(pixelColor[2]),0,0.999f);
	fb[pixel_index] = pixelColor;
}

#define RND (curand_uniform(&local_rand_state))	

__global__ void createCornellBox(hittable** d_list, hittable** d_world, camera** cam, int imageWidth, int imageHeight)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{

		defyPhysics = false;
		allowLightSource = true;

		int i = 0;
		auto red = new lambertian(glm::vec3(0.65, 0.05, 0.05));
		auto green = new lambertian(glm::vec3(0.12, 0.45, 0.15));
		auto light = new diffuse_light(glm::vec3(1));
		auto white = new lambertian(glm::vec3(0.73));
		auto magenta = new lambertian(glm::vec3(0.8, 0.2, 0.9));
		auto blue = new metal(glm::vec3(0.1, 0.2, 0.9), 0.5);

		d_list[i++] = new flip_face(new yz_rect(0, 555, 0, 555, 555, green));
		d_list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
		d_list[i++] = new flip_face(new xz_rect(113, 443, 127, 432, 554, light));
		d_list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
		d_list[i++] = new flip_face(new xz_rect(0, 555, 0, 555, 555, white));
		d_list[i++] = new flip_face(new xy_rect(0, 555, 0, 555, 555, white));

		//d_list[i++] = new sphere(glm::vec3(50, 10, 30), 10, white);
		
		d_list[i++] = new box(glm::vec3(130, 0, 65) , glm::vec3(295, 165, 230) , white);
		d_list[i++] = new box(glm::vec3(265, 0, 295), glm::vec3(430, 330, 460) , white);


		*d_world = new hittable_list(d_list, 8);

		float aperture = 0.0;
		float dist_to_focus = 10.0;

		//CAMERA
		*cam = new camera(glm::vec3(278, 278, -800),
			glm::vec3(278, 278, 0),
			glm::vec3(0, 1, 0), 40,
			float(imageWidth)/float(imageHeight));
	}

}

__global__ void debugScene(hittable** list, hittable** world, camera** cam, int imageWidth, int imageHeight) {
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{

		defyPhysics = false;
		allowLightSource = true;

		int i = 0;

		auto ground = new metal(glm::vec3(0.1f),0.1f);
		auto red = new lambertian(glm::vec3(1, 0, 0));
		auto light = new diffuse_light(glm::vec3(4));

		list[i++] = new sphere(glm::vec3(0, -1000, 0), 1000, ground);
		list[i++] = new sphere(glm::vec3(-4, 1, 0), 1, red);
		list[i++] = new sphere(glm::vec3( 0, 1, 0), 1, red);
		list[i++] = new sphere(glm::vec3( 4, 1, 0), 1, red);

		list[i++] = new xy_rect(-10, 10, 0, 10, -5 , light);

		*world = new hittable_list(list, i);


		//CAMERA
		*cam = new camera(glm::vec3(0, 2, 6),
			glm::vec3(0, 0, 0),
			glm::vec3(0, 1, 0), 90,
			float(imageWidth) / float(imageHeight));

	}
}

__global__ void scene1(hittable** d_list, hittable** d_world, camera** cam , int imageWidth, int imageHeight, int limit, curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		defyPhysics = false;
		allowLightSource = true;

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
					d_list[i++] = new sphere(centre, 0.2, new metal(glm::vec3(0.8,0.2,0.9),0.0f));
					//d_list[i++] = new sphere(centre, 0.2, new metal(glm::vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
					//d_list[i++] = new sphere(centre, 0.2, new lambertian(glm::vec3(RND * RND, RND * RND, RND * RND)));
					//d_list[i++] = new sphere(centre, 0.2, new lambertian(glm::vec3(RND, RND, RND)));
				}
				else {
					d_list[i++] = new sphere(centre, 0.2, new metal(glm::vec3(0.1,0.2,0.9),0.0f));
					//d_list[i++] = new sphere(centre, 0.2, new metal(glm::vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
				}
			}
		}

		d_list[i++] = new sphere(glm::vec3(0, -1000, 0), 1000, new metal(glm::vec3(0.1f),0.2f));

		d_list[i++] = new sphere(glm::vec3(-4, 1, 0), 1, new metal(glm::vec3(0.2, 0.9, 0.1),0.0f));
		d_list[i++] = new sphere(glm::vec3(0, 1, 0), 1, new metal(glm::vec3(0.2, 0.9, 0.1),0.0f));
		d_list[i++] = new sphere(glm::vec3(4, 1, 0), 1, new metal(glm::vec3(0.2, 0.9, 0.1), 0.0f));
		d_list[i++] = new xy_rect(-3, 3, 0, 4, -7, new diffuse_light(glm::vec3(2)));

		*rand_state = local_rand_state;


		*d_world = new hittable_list(d_list, 4 * limit * limit + 1 + 4);

		//CAMERA
		*cam = new camera(glm::vec3(0, 2, 7),
			glm::vec3(0, 2, 0),
			glm::vec3(0, 1, 0), 45,
			float(imageWidth) / float(imageHeight));
	}
}

__global__ void scene2(hittable** d_list, hittable** d_world, camera** cam, int imageWidth, int imageHeight, int limit, curandState* rand_state) {

	if(threadIdx.x == 0 && blockIdx.x == 0)
	{

		defyPhysics = false;
		allowLightSource = false;

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
					//d_list[i++] = new sphere(centre, 0.2, new alienMat(glm::vec3(0.1, 0.2, 0.9), 0.0f));
					d_list[i++] = new sphere(centre, 0.2, new metal(glm::vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)),0));
				}
			}
		}

		d_list[i++] = new sphere(glm::vec3(0, -1000, 0), 1000, new lambertian(glm::vec3(0.7,0.6,0.5)));

		d_list[i++] = new sphere(glm::vec3(-4, 1, 0), 1, new metal(glm::vec3(0.7, 0.6, 0.5),0.0f));
		d_list[i++] = new sphere(glm::vec3( 0, 1, 0), 1, new metal(glm::vec3(0.7, 0.6, 0.5),0.0f));
		d_list[i++] = new sphere(glm::vec3( 4, 1, 0), 1, new metal(glm::vec3(0.7, 0.6, 0.5),0.0f));
		//d_list[i++] = new xy_rect(-3, 3, 0, 4, -7, new diffuse_light(glm::vec3(4, 0, 4)));

		*rand_state = local_rand_state;


		*d_world = new hittable_list(d_list, 4 * limit * limit + 1 + 3);

		//CAMERA
		*cam = new camera(glm::vec3(0, 2, 8),
			glm::vec3(0, 2, 0),
			glm::vec3(0, 1, 0), 50,
			float(imageWidth) / float(imageHeight));
	}

}

__global__ void scene3(hittable** d_list, hittable** d_world, camera** cam, int imageWidth, int imageHeight, int limit, curandState* rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		defyPhysics = true;
		allowLightSource = true;

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
					d_list[i++] = new sphere(centre, 0.2, new alienMat(glm::vec3(0.8, 0.2, 0.9), 0.0f));
					//d_list[i++] = new sphere(centre, 0.2, new metal(glm::vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
					//d_list[i++] = new sphere(centre, 0.2, new lambertian(glm::vec3(RND * RND, RND * RND, RND * RND)));
					//d_list[i++] = new sphere(centre, 0.2, new lambertian(glm::vec3(RND, RND, RND)));
				}
				else {
					d_list[i++] = new sphere(centre, 0.2, new alienMat(glm::vec3(0.1, 0.2, 0.9), 0.0f));
					//d_list[i++] = new sphere(centre, 0.2, new metal(glm::vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
				}
			}
		}

		d_list[i++] = new sphere(glm::vec3(0, -1000, 0), 1000, new metal(glm::vec3(0.1f), 0.2f));

		d_list[i++] = new sphere(glm::vec3(-4, 1, 0), 1, new metal(glm::vec3(0.1), 0.0f));
		d_list[i++] = new sphere(glm::vec3( 0, 1, 0), 1, new metal(glm::vec3(0.1), 0.0f));
		d_list[i++] = new sphere(glm::vec3( 4, 1, 0), 1, new metal(glm::vec3(0.1), 0.0f));
		//d_list[i++] = new xy_rect(-3, 3, 0, 4, -7, new diffuse_light(glm::vec3(2)));

		*rand_state = local_rand_state;


		*d_world = new hittable_list(d_list, 4 * limit * limit + 1 + 3);

		//CAMERA
		*cam = new camera(glm::vec3(0, 2, 7),
			glm::vec3(0, 2, 0),
			glm::vec3(0, 1, 0), 45,
			float(imageWidth) / float(imageHeight));
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

	//stbi_flip_vertically_on_write(1);

	const float aspectRatio = 16.0f / 9.0f;//3840.0f / 2160.0f;
	const int imageWidth = 1600;
	const int imageHeight = imageWidth / aspectRatio;
	int num_of_pixels = imageWidth * imageHeight;
	int number_of_samples = 10;
	int depth = 10;
	int limit = 3;
	int tx = 16;
	int ty = 16;

	std::cerr << "Rendering a " << imageWidth << "x" << imageHeight << " image \n";
	std::cerr << "Number of blocks: " << (imageWidth / tx + 1) * (imageHeight / ty + 1) << "\n";
	std::cerr << "Number of threads per block: " << tx * ty << "\n";
	std::cerr << "Samples per pixel: " << number_of_samples << "\n";
	std::cerr << "Max depth of ray bounce: " << depth << "\n";

	size_t fb_size = num_of_pixels * sizeof(glm::vec3);
	glm::vec3* fb;
	CudaCall(cudaMallocManaged((void**)&fb, fb_size));

	curandState* d_rand_state;
	CudaCall(cudaMalloc((void**)&d_rand_state, num_of_pixels * sizeof(curandState)));
	curandState* d_rand_state2;
	CudaCall(cudaMalloc((void**)&d_rand_state2, num_of_pixels * sizeof(curandState)));

	rand_init <<<1,1>>> (d_rand_state2);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());

	int choice = 0;

	switch (choice)
	{
	case 1 : 
		break;
	case 2 : 
		break;
	case 3 :
		break;
	case 4 :
		break;
	default :
		break;
	}

	//WORLD
	hittable** d_list;
	CudaCall(cudaMalloc((void**)&d_list, (4 * limit * limit + 1 + 4) * sizeof(hittable*)));
	hittable** d_world;
	CudaCall(cudaMalloc((void**)&d_world, sizeof(hittable*)));
	camera** cam;
	CudaCall(cudaMalloc((void**)&cam, sizeof(camera*)));
	scene1 << <1, 1 >> > (d_list, d_world, cam, imageWidth, imageHeight, limit,d_rand_state2);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());


	dim3 blocks(imageWidth / tx + 1, imageHeight / ty + 1);
	dim3 threads(tx, ty);
	render_init <<<blocks,threads>>> (imageWidth, imageHeight, d_rand_state);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());

	auto start = clock();

	render <<<blocks, threads >> > (fb, imageWidth, imageHeight, cam, d_world, d_rand_state, number_of_samples, depth);
	CudaCall(cudaGetLastError());
	CudaCall(cudaDeviceSynchronize());

	auto stop = clock();

	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "Took " << timer_seconds << " seconds.\n";

	uint8_t* imageHost = new uint8_t[imageWidth * imageHeight * 3 * sizeof(uint8_t)];
	for (int j = imageHeight - 1; j >= 0; j--) {
		for (int i = 0; i < imageWidth; i++) {
			size_t pixel_index = j * imageWidth + i;
			imageHost[(imageHeight - j - 1) * imageWidth * 3 + i * 3] = 255.99 * fb[pixel_index].r();
			imageHost[(imageHeight - j - 1) * imageWidth * 3 + i * 3 + 1] = 255.99 * fb[pixel_index].g();
			imageHost[(imageHeight - j - 1) * imageWidth * 3 + i * 3 + 2] = 255.99 * fb[pixel_index].b();
		}
	}


	int result = stbi_write_png("res/images/output.png", imageWidth, imageHeight, 3, imageHost, imageWidth * 3);
	std::cout << "PNG Created : " << result << "\n";

	// clean up
	CudaCall(cudaDeviceSynchronize());
	CudaCall(cudaGetLastError());
	CudaCall(cudaFree(cam));
	CudaCall(cudaFree(d_world));
	CudaCall(cudaFree(d_list));
	CudaCall(cudaFree(d_rand_state));
	//CudaCall(cudaFree(d_rand_state2));
	CudaCall(cudaFree(fb));

}

