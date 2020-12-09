#include "neat.h"

namespace neat {

NEAT::NEAT() {}

void NEAT::Clear() {
    config_.Clear();
    gene_pool_.Clear();
    species_.clear();
    genotypes_.clear();
}

void NEAT::Initialize(const unsigned int& n_sensor_nodes, const unsigned int& n_output_nodes,
                      const unsigned int& n_genotypes, const Config config) {
    Clear();
    config_ = config;
    n_genotypes_init_ = n_genotypes;
    gene_pool_.Initialize(n_sensor_nodes, n_output_nodes);
    genotypes_.resize(n_genotypes);

    for (size_t i = 0; i < genotypes_.size(); i++) {
        for (size_t j = 0; j < gene_pool_.GetGenes().size(); j++) {
            genotypes_.at(i).genes.push_back(
                genome::Gene(j, random_.RandomNumber(config_.weight_range.first, config_.weight_range.second)));
        }

        for(size_t j= 0; j < gene_pool_.GetNTotalNodes(); j++){
            genotypes_.at(i).nodes.push_back(j);
        }
    }

    neat_algorithms::SortInSpecies(genotypes_, species_, config_.species_distance, config_.coeff1, config_.coeff2,
                                   config_.coeff3);
}

std::vector<std::vector<double>> NEAT::ExecuteNetworks(const std::vector<double>& input_values) const {
    std::vector<std::vector<double>> results(genotypes_.size());

    if (input_values.size() != gene_pool_.GetNSensorNodes()) {
        throw std::domain_error("number input values != sensor nodes");
    }

    for (uint i = 0; i < genotypes_.size(); i++) {
        results.at(i) = ExecuteNetwork(input_values, i);
    }

    return results;
}

std::vector<double> NEAT::ExecuteNetwork(const std::vector<double>& input_values, const uint& genotype_id) const {
    std::vector<double> retvals(gene_pool_.GetNOutputNodes(), 0.0);
    VectorXd nodes(neat_algorithms::SetUpNodes(input_values, gene_pool_));
    const MatrixXd matrix(neat_algorithms::Genotype2Phenotype(genotypes_.at(genotype_id), gene_pool_));
    const uint n_const_nodes(gene_pool_.GetNSensorNodes() + 1);

    for (uint i = 0; i < genotypes_.at(genotype_id).nodes.size(); i++) {
        neat_algorithms::ExecuteNetwork(matrix, nodes, n_const_nodes, config_.sigmoid_parameter);
    }

    for (uint i = n_const_nodes, j = 0; i < n_const_nodes + gene_pool_.GetNOutputNodes(); i++, j++) {
        retvals.at(j) = nodes(i);
    }

    return retvals;
}

void NEAT::UpdateNetworks(std::vector<double> fitnesses) {
    neat_algorithms::AdjustedFitnesses(fitnesses, species_, genotypes_);
    neat_algorithms::Reproduce(fitnesses, species_, genotypes_, n_genotypes_init_);
    // mutate
    neat_algorithms::SortInSpecies(genotypes_, species_, config_.species_distance, config_.coeff1, config_.coeff2,
                                   config_.coeff3);
}

GenePool NEAT::GetGenePool() const { return gene_pool_; }

std::vector<genome::Genotype> NEAT::GetGenotypes() const { return genotypes_; }

std::vector<genome::Species> NEAT::GetSpecies() const { return species_; }

void NEAT::SetGenotypes(const std::vector<genome::Genotype>& genotypes) { genotypes_ = genotypes; }

}  // namespace neat
