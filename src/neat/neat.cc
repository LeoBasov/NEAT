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
    const uint max(std::max(fitter_parent.genes_.back().id, less_fit_parent.genes_.back().id));

    for (uint i = 0, fit = 0, less = 0; i < max; i++) {
        if (fitter_parent.genes_.at(fit).id == i && less_fit_parent.genes_.at(less).id == i) {
            if (random_.RandomNumber() > 0.5) {
                child.AddGene(fitter_parent.genes_.at(fit), gene_pool_.genes_.at(fitter_parent.genes_.at(fit).id).in, gene_pool_.genes_.at(fitter_parent.genes_.at(fit).id).out);
                fit++;
                less++;
            } else {
                child.AddGene(less_fit_parent.genes_.at(fit), gene_pool_.genes_.at(less_fit_parent.genes_.at(fit).id).in, gene_pool_.genes_.at(less_fit_parent.genes_.at(fit).id).out);
                fit++;
                less++;
            }
        } else if (fitter_parent.genes_.at(fit).id == i) {
            child.AddGene(fitter_parent.genes_.at(fit), gene_pool_.genes_.at(fitter_parent.genes_.at(fit).id).in, gene_pool_.genes_.at(fitter_parent.genes_.at(fit).id).out);
            fit++;
        }
    }

    return child;
}

void NEAT::GenerateNetworks() {
    networks_.clear();

    for (uint i = 0; i < phenotypes_.size(); i++) {
        networks_.push_back(GenerateNetwork(phenotypes_.at(i)));
    }
}

Network NEAT::GenerateNetwork(const Phenotype& phenotype) const {
    Network network(gene_pool_.depth_);

    for (uint level = 1; level < gene_pool_.depth_; level++) {
        MatrixXd matrix(gene_pool_.node_ofset_.at(level), gene_pool_.node_ofset_.at(level - 1));

        for (uint node_id_outer = gene_pool_.node_ofset_.at(level), i = 0;
             node_id_outer < gene_pool_.node_ofset_.at(level + 1); node_id_outer++, i++) {
            for (uint node_id_inner = gene_pool_.node_ofset_.at(level - 1), j = 0;
                 node_id_inner < gene_pool_.node_ofset_.at(level); node_id_inner++, j++) {
                if (phenotype.HasNode(node_id_outer)) {
                    for (uint gene_id = 0; gene_id < phenotype.genes_.size(); gene_id++) {
                        if ((gene_pool_.genes_.at(phenotype.genes_.at(i).id).in == node_id_inner) &&
                            (gene_pool_.genes_.at(phenotype.genes_.at(i).id).out == node_id_outer)) {
                            if (phenotype.genes_.at(i).enabled) {
                                matrix(i, j) = phenotype.genes_.at(i).weight;
                            } else {
                                matrix(i, j) = 0.0;
                            }
                        }
                    }
                } else {
                    matrix(i, j) = 0.0;
                }
            }
        }

        network.level_matrizes_.at(level) = matrix;
    }
}

}  // namespace NEAT
