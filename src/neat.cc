#include "neat.h"

namespace NEAT {

NEAT::NEAT() {}

void NEAT::Clear() {
    gene_pool_.Clear();
    phenotypes_.clear();
    networks_.clear();
}

void NEAT::Execute(const std::vector<std::pair<VectorXd, VectorXd>> &input_outputs) {
    std::vector<double> fitness_vec(phenotypes_.size());
    VectorXd output;

    for (uint outer_id = 0; outer_id < input_outputs.size(); outer_id++) {
        for (uint inner_id = 0; inner_id < networks_.size(); inner_id++) {
            networks_.at(inner_id).Execute(input_outputs.at(outer_id).first, output);
            fitness_vec.at(inner_id) = (input_outputs.at(outer_id).second - output).norm();
        }
    }

    for (uint i = 0; i < networks_.size(); i++) {
        phenotypes_.at(i).fitness_ = fitness_vec.at(i) / static_cast<double>(input_outputs.size());
    }
}

}  // namespace NEAT
