#pragma once

#include <Eigen/Dense>
#include <vector>

namespace NEAT {
using namespace Eigen;
class Network {
   public:
    Network(const uint& depth = 0);
    ~Network() = default;

    void Execute(const VectorXd& input, VectorXd output) const {
        for (uint i = 0; i < level_matrizes_.size(); i++) {
            if (i == 0) {
                output = level_matrizes_.at(i) * input;
            } else {
                output = level_matrizes_.at(i) * output;
            }

            // TODO: #pragma omp parallel for
            for (uint j = 0; j < output.rows(); j++) {
                output(j) = 1.0 / (1.0 + std::exp(-output(j)));
            }
        }
    }

    std::vector<MatrixXd> level_matrizes_;
};
}  // namespace NEAT
