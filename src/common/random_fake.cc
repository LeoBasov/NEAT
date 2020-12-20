#include "random_fake.h"

namespace neat {

RandomFake::RandomFake() {}

void RandomFake::SetRetVal(double ret_val) { ret_val_ = ret_val; }

void RandomFake::SetRetValInt(int ret_val_int) { ret_val_int_ = ret_val_int; }

double RandomFake::RandomNumber(const double&, const double&) { return ret_val_; }

int RandomFake::RandomIntNumber(const int &min, const int &max) { return ret_val_int_; }

Vector3d RandomFake::RandomVector(Vector3d, Vector3d) { return Vector3d(0.0, 0.0, 0.0); }

double RandomFake::NormalRandomNumber(const double&, const double&) { return ret_val_; }

Vector3d RandomFake::NormalRandomVector(const double&, const double&) { return Vector3d(0.0, 0.0, 0.0); }

}  // namespace neat
