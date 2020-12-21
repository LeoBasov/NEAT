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

    void Build(const Genome& genome, bool cyclic_check = true);
    std::vector<double> Execute(const std::vector<double>& input_vaules);

    static std::vector<size_t> SetUpOutputNodes(const Genome& genome);
    static MatrixXd GenomeToMatrix(const Genome& genome);
    static VectorXd GenomeToVector(const std::vector<double>& input_vaules, const uint& n_nodes);
    static std::vector<double> VectorToStd(const VectorXd& vec, const std::vector<size_t>& output_nodes);

    MatrixXd matrix_;
    std::vector<size_t> output_nodes_;
    uint n_executions_ = 0;
    uint n_const_nodes_ = 0;
    bool cyclic_check_ = true, cyclic_ = false;
    double parameter_ = 5.9;
};
}  // namespace neat
