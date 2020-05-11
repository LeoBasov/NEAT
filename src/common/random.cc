#include "random.h"

namespace NEAT {
Random::Random() {}

double Random::RandomNumber(const double& min, const double& max) {
    std::uniform_real_distribution<double> uniform_distribution{min, max};
    return uniform_distribution(mersenne_twister_);
}

Vector3d Random::RandomVector(Vector3d min, Vector3d max) {
    return Vector3d(RandomNumber(min(0), max(0)), RandomNumber(min(1), max(1)), RandomNumber(min(2), max(2)));
}

double Random::NormalRandomNumber(const double& mean, const double& stddev) {
    std::normal_distribution<double> distribution(mean, stddev);
    return distribution(mersenne_twister_);
}

Vector3d Random::NormalRandomVector(const double& mean, const double& stddev) {
    return Vector3d(NormalRandomNumber(mean, stddev), NormalRandomNumber(mean, stddev),
                    NormalRandomNumber(mean, stddev));
}

}  // namespace NEAT
