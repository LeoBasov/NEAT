#include "neat.h"

namespace NEAT {

NEAT::NEAT() {}

void NEAT::Clear() {
    gene_pool_.Clear();
    phenotypes_.clear();
    networks_.clear();
    species_.clear();
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
            fitness_vec.at(inner_id) += 1.0 - (input_outputs.at(outer_id).second - output).norm();
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
        if (in_weight.first < gene_pool_.input_nodes_.n_parts) {
            ret_val += in_weight.second * input(in_weight.first);
        } else {
            ret_val += in_weight.second * ExecuteNode(network_id, in_weight.first, input);
        }
    }

    return Sigmoid(ret_val, config_.sigmoid_parameter);
}

Phenotype NEAT::Mate(const Phenotype& fitter_parent, const Phenotype& less_fit_parent) {
    Phenotype child;
    const uint max(std::max(fitter_parent.genes_.back().id, less_fit_parent.genes_.back().id));

    for (uint i = 0, fit = 0, less = 0; i <= max; i++) {
        if (((fit < fitter_parent.genes_.size()) && (less < less_fit_parent.genes_.size())) &&
            fitter_parent.genes_.at(fit).id == i && less_fit_parent.genes_.at(less).id == i) {
            if (random_.RandomNumber() > 0.5) {
                child.AddGene(fitter_parent.genes_.at(fit));
                fit++;
                less++;
            } else {
                child.AddGene(less_fit_parent.genes_.at(less));
                fit++;
                less++;
            }
        } else if ((fit < fitter_parent.genes_.size()) && (fitter_parent.genes_.at(fit).id == i)) {
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

double NEAT::Sigmoid(const double& value, const double& paramters) const {
    return 1.0 / (1.0 + std::exp(-paramters * value));
}

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

        for (auto& gene : phenotypes_.at(phenotype_id).genes_) {
            GenePool::Gene gene_p(gene_pool_.genes_.at(gene.id));

            if ((gene_p.in == in) && (gene_p.out == out)) {
                weight = gene.weight;
                gene.enabled = false;
                break;
            }
        }

        phenotypes_.at(phenotype_id).AddGene(gene_pool_.genes_.size() - 2);
        phenotypes_.at(phenotype_id).AddGene(gene_pool_.genes_.size() - 1);
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

double NEAT::Distance(const Phenotype& first, const Phenotype& second, const std::array<double, 3>& parameters) const {
    double exess(0.0), disjoint(0.0), weights(0.0), tot_weigh_n(0.0);
    const uint max_id(std::max(first.genes_.back().id, second.genes_.back().id));
    const double max_size(std::max(first.genes_.size(), second.genes_.size()));
    const uint small_max(std::min(first.genes_.back().id, second.genes_.back().id));

    for (uint i = 0, f = 0, s = 0; i <= max_id; i++) {
        if (((f < first.genes_.size()) && (s < second.genes_.size())) && first.genes_.at(f).id == i &&
            second.genes_.at(s).id == i) {
            weights += std::abs(first.genes_.at(f).weight - second.genes_.at(s).weight);
            tot_weigh_n += 1.0;

            f++;
            s++;
        } else if ((f < first.genes_.size()) && (first.genes_.at(f).id == i)) {
            if (first.genes_.at(f).id > small_max) {
                exess += 1.0;
            } else {
                disjoint += 1.0;
            }

            f++;
        } else if ((s < second.genes_.size()) && (second.genes_.at(s).id == i)) {
            if (second.genes_.at(f).id > small_max) {
                exess += 1.0;
            } else {
                disjoint += 1.0;
            }

            s++;
        }
    }

    return (parameters.at(0) * exess / max_size) + (parameters.at(1) * disjoint / max_size) +
           (parameters.at(2) * weights / tot_weigh_n);
}

void NEAT::Speciate() {
    double total_adjuste_fitness(0.0);

    for (auto& spcies : species_) {
        spcies.Clear();
    }

    for (uint i = 0; i < phenotypes_.size(); i++) {
        bool found(false);

        for (auto& spcies : species_) {
            const double distance(
                Distance(phenotypes_.at(i), spcies.ref_phenotype, config_.species_distance_parameters));

            if (distance < config_.species_distance_threshold) {
                spcies.phenotype_ids.push_back(i);
                found = true;
                break;
            }
        }

        if (!found) {
            Species species(phenotypes_.at(i));
            species.phenotype_ids.push_back(i);
            species_.push_back(species);
        }
    }

    for (auto& species : species_) {
        species.ref_phenotype = phenotypes_.at(species.phenotype_ids.front());

        for (const auto& id : species.phenotype_ids) {
            phenotypes_.at(id).fitness_ /= species.phenotype_ids.size();
            species.total_adjusted_fitness += phenotypes_.at(id).fitness_;
            total_adjuste_fitness += phenotypes_.at(id).fitness_;
        }
    }

    for (auto& species : species_) {
        species.n_offspring = (species.total_adjusted_fitness / total_adjuste_fitness) * config_.n_phenotypes;
    }
}

void NEAT::Reproduce() {}

}  // namespace NEAT
