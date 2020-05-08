#pragma once

#include <Eigen/Dense>
#include <vector>

namespace NEAT {
using namespace Eigen;
class Network {
    Network();
    ~Network() = default;

    void Execute(const VectorXd& input, VectorXd output) const {
        for (uint i = 0; i < level_matrizes_.size(); i++) {
            output = level_matrizes_.at(i) * input;

            // TODO: #pragma omp parallel for
            for (uint j = 0; j < output.rows(); j++) {
                output(j) = 1.0 / (1.0 + std::exp(-output(j)));
            }
        }
    }

   private:
    std::vector<MatrixXd> level_matrizes_;
};
}  // namespace NEAT
