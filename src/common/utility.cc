#include "utility.h"

namespace neat {
namespace utility {

double Sigmoid(const double& value, const double& parameter) { return 1.0 / (1.0 + std::exp(-parameter * value)); }

}  // namespace utility
}  // namespace neat
