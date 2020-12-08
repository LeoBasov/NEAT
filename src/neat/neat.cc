#include "neat.h"

namespace neat {

NEAT::NEAT() {}

void NEAT::Clear() {
    config_.Clear();
    gene_pool_.Clear();
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
                Gene(j, random_.RandomNumber(config_.weight_range.first, config_.weight_range.second)));
        }
    }
}

GenePool NEAT::GetGenePool() const { return gene_pool_; }

std::vector<NEAT::Genotype> NEAT::GetGenotypes() const { return genotypes_; }

}  // namespace neat
