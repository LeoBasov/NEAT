#pragma once

#include <Eigen/Dense>
#include <map>
#include <vector>

namespace NEAT {
using namespace Eigen;
class Network {
   public:
    struct Node {
        std::vector<std::pair<uint, double>> in_weights;
    };

   public:
    Network();
    ~Network() = default;

    std::map<uint, Node> nodes_;
};
}  // namespace NEAT
