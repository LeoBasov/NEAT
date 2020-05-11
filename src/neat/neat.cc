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
    VectorXd output(gene_pool_.output_nodes_.n_parts);

    for (uint outer_id = 0; outer_id < input_outputs.size(); outer_id++) {
        for (uint inner_id = 0; inner_id < phenotypes_.size(); inner_id++) {
            ExecuteNetwork(inner_id, input_outputs.at(outer_id).first, output);
            fitness_vec.at(inner_id) = (input_outputs.at(outer_id).second - output).norm();
        }
    }

    for (uint i = 0; i < phenotypes_.size(); i++) {
        phenotypes_.at(i).fitness_ = fitness_vec.at(i) / static_cast<double>(input_outputs.size());
    }
}

void NEAT::ExecuteNetwork(const uint& network_id, const VectorXd& input, VectorXd& output) const {}

Phenotype NEAT::Mate(const Phenotype& fitter_parent, const Phenotype& less_fit_parent) {
    Phenotype child;
    const uint max(std::max(fitter_parent.genes_.back().id, less_fit_parent.genes_.back().id));

    for (uint i = 0, fit = 0, less = 0; i < max; i++) {
        if (fitter_parent.genes_.at(fit).id == i && less_fit_parent.genes_.at(less).id == i) {
            if (random_.RandomNumber() > 0.5) {
                child.AddGene(fitter_parent.genes_.at(fit));
                fit++;
                less++;
            } else {
                child.AddGene(less_fit_parent.genes_.at(fit));
                fit++;
                less++;
            }
        } else if (fitter_parent.genes_.at(fit).id == i) {
            child.AddGene(fitter_parent.genes_.at(fit));
            fit++;
        }
    }

    return child;
}

void NEAT::BuildNetworks() {
    networks_.cend();

    for (auto phenotype : phenotypes_) {
        Network network;

        for (auto gene : phenotype.genes_) {
            if (gene.enabled) {
                GenePool::Gene pool_gene(gene_pool_.genes_.at(gene.id));

                if (network.nodes_.count(pool_gene.out)) {
                    network.nodes_[pool_gene.out].in_weights.push_back({pool_gene.in, gene.weight});
                } else {
                    network.nodes_[pool_gene.out] = Network::Node();
                    network.nodes_[pool_gene.out].in_weights.push_back({pool_gene.in, gene.weight});
                }
            }
        }
    }
}

}  // namespace NEAT
