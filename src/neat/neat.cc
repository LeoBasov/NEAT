#include "neat.h"

namespace NEAT {

NEAT::NEAT() {}

void NEAT::Clear() {
    gene_pool_.Clear();
    phenotypes_.clear();
    networks_.clear();
}

void NEAT::Execute(const std::vector<std::pair<VectorXd, VectorXd>>& input_outputs) {
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

Phenotype NEAT::Mate(const Phenotype& fitter_parent, const Phenotype& less_fit_parent) {
    Phenotype child;
    const uint max(std::max(fitter_parent.genes.back().id, less_fit_parent.genes.back().id));

    for (uint i = 0, fit = 0, less = 0; i < max; i++) {
        if (fitter_parent.genes.at(fit).id == i && less_fit_parent.genes.at(less).id == i) {
            if (random_.RandomNumber() > 0.5) {
                child.genes.push_back(fitter_parent.genes.at(fit++));
                less++;
            } else {
                child.genes.push_back(less_fit_parent.genes.at(less++));
                fit++;
            }
        } else if (fitter_parent.genes.at(fit).id == i) {
            child.genes.push_back(fitter_parent.genes.at(fit++));
        }
    }

    return child;
}

}  // namespace NEAT
