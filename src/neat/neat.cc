#include "neat.h"

namespace NEAT {

NEAT::NEAT() {}

void NEAT::Clear() {
    gene_pool_.Clear();
    phenotypes_.clear();
    networks_.clear();
}

void NEAT::Initialize(const Config& config) {
    Clear();
    config_ = config;
    gene_pool_.Initialize(config_.n_input, config_.n_output);
    phenotypes_.resize(config_.n_phenotypes);

    for (uint i = 0; i < gene_pool_.genes_.size(); i++) {
        for (uint j = 0; j < phenotypes_.size(); j++) {
            phenotypes_.at(j).AddGene(i);
        }
    }
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

void NEAT::ExecuteNetwork(const uint& network_id, const VectorXd& input, VectorXd& output) const {
    for (uint i = gene_pool_.output_nodes_.ofset, j = 0;
         i < gene_pool_.output_nodes_.ofset + gene_pool_.output_nodes_.n_parts; i++, j++) {
        output(j) = ExecuteNode(network_id, i, input);
    }
}

double NEAT::ExecuteNode(const uint& network_id, const uint& node_id, const VectorXd& input) const {
    double ret_val(0.0);

    for (auto in_weight : networks_.at(network_id).nodes_.at(node_id).in_weights) {
        if (in_weight.second < gene_pool_.input_nodes_.n_parts) {
            ret_val += input(in_weight.second);
        } else {
            ret_val += in_weight.first * ExecuteNode(network_id, in_weight.second, input);
        }
    }

    return Sigmoid(ret_val);
}

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
    networks_.resize(phenotypes_.size());

    for (uint i = 0; i < phenotypes_.size(); i++) {
        networks_.at(i).nodes_.clear();

        for (auto gene : phenotypes_.at(i).genes_) {
            if (gene.enabled) {
                GenePool::Gene pool_gene(gene_pool_.genes_.at(gene.id));

                if (networks_.at(i).nodes_.count(pool_gene.out)) {
                    networks_.at(i).nodes_[pool_gene.out].in_weights.push_back({pool_gene.in, gene.weight});
                } else {
                    networks_.at(i).nodes_[pool_gene.out] = Network::Node();
                    networks_.at(i).nodes_[pool_gene.out].in_weights.push_back({pool_gene.in, gene.weight});
                }
            }
        }
    }
}

double NEAT::Sigmoid(const double& value) const { return 1.0 / (1.0 + std::exp(-value)); }

bool NEAT::AddConnection(const uint& phenotype_id, const uint& in, const uint& out) {
    auto ret_val = gene_pool_.AddConnection(in, out);

    if (ret_val.first) {
        return phenotypes_.at(phenotype_id).AddGeneWithCheck(ret_val.second);
    } else {
        return false;
    }
}

bool NEAT::AddNode(const uint& phenotype_id, const uint& in, const uint& out) {
    if (gene_pool_.AddNode(in, out)) {
        double weight(0.0);

        for (auto gene : phenotypes_.at(phenotype_id).genes_) {
            GenePool::Gene gene_p(gene_pool_.genes_.at(gene.id));

            if ((gene_p.in == in) && (gene_p.out == out)) {
                weight = gene.weight;
                gene.enabled = false;
                break;
            }
        }

        phenotypes_.at(phenotype_id).AddGene(gene_pool_.genes_.size() - 1);
        phenotypes_.at(phenotype_id).AddGene(gene_pool_.genes_.size() - 2);
        phenotypes_.at(phenotype_id).genes_.at(phenotypes_.at(phenotype_id).genes_.size() - 1).weight = weight;
    } else {
        return false;
    }
}

bool NEAT::SetWeight(const uint& phenotype_id, const uint& gene_id, const double& weight) {
    for (auto& gene : phenotypes_.at(phenotype_id).genes_) {
        if (gene.id == gene_id) {
            gene.weight = weight;
            return true;
        }
    }

    return false;
}

bool NEAT::ChangeActivation(const uint& phenotype_id, const uint& gene_id) {
    for (auto& gene : phenotypes_.at(phenotype_id).genes_) {
        if (gene.id == gene_id) {
            gene.enabled = !gene.enabled;
            return true;
        }
    }

    return false;
}

}  // namespace NEAT
