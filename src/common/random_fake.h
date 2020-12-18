#pragma once

#include "random.h"

namespace neat {
class RandomFake : public Random {
   public:
    RandomFake();
    ~RandomFake() = default;

    void SetRetVal(double ret_val);

    double RandomNumber(const double &min = 0.0, const double &max = 1.0) override;
    Vector3d RandomVector(Vector3d min = Vector3d(0.0, 0.0, 0.0), Vector3d max = Vector3d(1.0, 1.0, 1.0)) override;
    double NormalRandomNumber(const double &mean = 0.0, const double &stddev = 1.0) override;
    Vector3d NormalRandomVector(const double &mean = 0.0, const double &stddev = 1.0) override;

   private:
    double ret_val_ = 0.0;
};
}  // namespace neat
