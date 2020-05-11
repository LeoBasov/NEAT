#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <random>

namespace NEAT {
using namespace Eigen;
class Random {
   public:
    Random();
    ~Random() = default;

    double RandomNumber(const double &min = 0.0, const double &max = 1.0);
    Vector3d RandomVector(Vector3d min = Vector3d(0.0, 0.0, 0.0), Vector3d max = Vector3d(1.0, 1.0, 1.0));
    double NormalRandomNumber(const double &mean = 0.0, const double &stddev = 1.0);
    Vector3d NormalRandomVector(const double &mean, const double &stddev);

   private:
    std::mt19937 mersenne_twister_ = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
};
}  // namespace NEAT
