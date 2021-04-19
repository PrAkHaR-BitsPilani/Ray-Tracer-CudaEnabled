//#include "rtweekend.h"
//
//#include "camera.h"
//#include "color.h"
//#include "hittable_list.h"
//#include "material.h"
//#include "sphere.h"
//
//#include "stb_image/stb_image.h"
//#include "stb_image/stb_image_write.h"
//
//#include <iostream>
//#include <vector>
//
//
//glm::vec3 ray_color(const ray& r, const hittable& world , int depth)
//{
//
//    if (depth <= 0)
//        return glm::vec3(0.0f);
//
//    hit_record rec;
//    if (world.hit(r, 0.001, infinity, rec))
//    {
//        ray scattered;
//        glm::vec3 attenuation;
//        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
//        {
//            return attenuation * ray_color(scattered, world, depth - 1);
//        }
//        return glm::vec3(0.0f);
//
//    }
//    glm::vec3 res;
//    glm::vec3 dir = glm::normalize(r.dir);
//    auto t = 0.5f * (dir.y + 1.0f);
//    res = (1.0f - t) * glm::vec3(1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
//    return res;
//}
//
//hittable_list random_scene() {
//    hittable_list world;
//
//    auto ground_material = new lambertian(glm::vec3(0.5, 0.5, 0.5));
//    world.add(new sphere(glm::vec3(0,-1000,0), 1000, ground_material));
//
//    int limit = 0;
//
//    for (int a = -limit; a < limit; a++) {
//        for (int b = -limit; b < limit; b++) {
//            auto choose_mat = random_float();
//            glm::vec3 center(a + 0.9*random_float(), 0.2, b + 0.9*random_float());
//
//            if ((center - glm::vec3(4, 0.2, 0)).length() > 0.9) {
//                material* sphere_material;
//
//                if (choose_mat < 0.8) {
//                    // diffuse
//                    auto albedo = randomColor() * randomColor();
//                    sphere_material = new lambertian(albedo);
//                    world.add(new sphere(center, 0.2, sphere_material));
//                } else {
//                    // metal
//                    auto albedo = randomColor();
//                    auto fuzz = random_float(0, 0.5);
//                    sphere_material = new metal(albedo, fuzz);
//                    world.add(new sphere(center, 0.2, sphere_material));
//                }
//            }
//        }
//    }
//
//    /*auto material2 = new lambertian(glm::vec3(0.4, 0.2, 0.1));
//    world.add(new sphere(glm::vec3(-4, 1, 0), 1.0, material2));*/
//
//    auto material3 = new metal(glm::vec3(0.7, 0.6, 0.5), 0.0);
//    world.add(new sphere(glm::vec3(4, 1, 0), 1.0, material3));
//    world.add(new sphere(glm::vec3(6, 1, 0), 1.0, material3));
//    world.add(new sphere(glm::vec3(8, 1, 0), 1.0, material3));
//    world.add(new sphere(glm::vec3(10, 1, 0), 1.0, material3));
//    world.add(new sphere(glm::vec3(12, 1, 0), 1.0, material3));
//
//    return world;
//}
//
//int  fakeMain()
//{
//
//    //Image
//    const auto aspectRatio = 16.0f / 9.0f;
//    const int imgWidth = 1600;
//    const int imgHeight = (int)(imgWidth/aspectRatio);
//    const int number_of_samples = 10;
//    const int maxDepth = 5;
//
//    //World
//    hittable_list world = random_scene();
//
//    //Camera
//    camera cam(glm::vec3(8, 2, 10), glm::vec3(8, 1, 0), glm::vec3(0, 1, 0), 45, aspectRatio);
//
//
//    
//
//    std::vector<unsigned char> img;
//
//    for (int j = imgHeight - 1; j >= 0; --j) {
//        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
//        for (int i = 0; i < imgWidth; ++i) {
//
//            glm::vec3 pixel_color(0.0f);
//            for (int s = 0; s < number_of_samples; s++)
//            {
//                auto u = float(i + random_float()) / (imgWidth - 1);
//                auto v = float(j + random_float()) / (imgHeight - 1);
//                ray r = cam.get_ray(u, v);
//                pixel_color += ray_color(r, world, maxDepth);
//            }
//            img.push_back(255 * clamp(sqrt(pixel_color.r/number_of_samples), 0, 0.999f));
//            img.push_back(255 * clamp(sqrt(pixel_color.g/number_of_samples), 0, 0.999f));
//            img.push_back(255 * clamp(sqrt(pixel_color.b/number_of_samples), 0, 0.999f));
//            
//
//        }
//    }
//
//    createJPG("res/images/output3.jpg", imgWidth, imgHeight, 3, img, 100);
//    std::cerr << "\nDone.\n";
//    return 0;
//
//}