#include "neat.h"

namespace NEAT {

NEAT::NEAT() {}

void NEAT::Clear() {
    gene_pool_.Clear();
    phenotypes_.clear();
    networks_.clear();
}

void NEAT::Execute(const std::pair<VectorXd, VectorXd> &input_output) {
    VectorXd output;

    for (uint i = 0; i < networks_.size(); i++) {
        networks_.at(i).Execute(input_output.first, output);
        phenotypes_.at(i).fitness_ = (input_output.second - output).norm();
    }
}

}  // namespace NEAT
