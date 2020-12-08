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

GenePool NEAT::GetGenePool() const { return gene_pool_; }

std::vector<genome::Genotype> NEAT::GetGenotypes() const { return genotypes_; }

std::vector<genome::Species> NEAT::GetSpecies() const { return species_; }

}  // namespace neat
