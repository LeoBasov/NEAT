#pragma once

#include <Eigen/Dense>

#include "../common/utility.h"
#include "genome.h"

namespace neat {
using namespace Eigen;
class Network {
   public:
    Network();
    Network(const Genome& genome);
    ~Network() = default;

    void Build(const Genome& genome);
    std::vector<double> Execute(const std::vector<double>& input_vaules);

    std::vector<size_t> SetUpOutputNodes(const Genome& genome) const;
    MatrixXd GenomeToMatrix(const Genome& genome) const;
    VectorXd GenomeToVector(const std::vector<double>& input_vaules, const uint& n_nodes) const;
    std::vector<double> VectorToStd(const VectorXd& vec) const;

    MatrixXd matrix_;
    std::vector<size_t> output_nodes_;
    uint n_executions_ = 0;
    uint n_const_nodes_ = 0;
    bool cyclic = false;
    double parameter_ = 4.9;
};
}  // namespace neat
