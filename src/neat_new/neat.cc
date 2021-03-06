#include "neat.h"

namespace neat {

Neat::Neat() {}

void Neat::Initialize(const uint& n_sensor_nodes, const uint& n_output_nodes, const uint& n_genotypes,
                      const Config config) {
    mutator_.SetConfig(config.mutator_config);
    species_pool_.SetConfig(config.species_pool_config);

    genomes_.clear();
    species_pool_.Clear();
    genomes_.resize(n_genotypes);

    for (uint i = 0; i < n_genotypes; i++) {
        genomes_.at(i).Initialize(n_sensor_nodes, n_output_nodes);

        for (uint k = 0; k < genomes_.at(i).genes_.size(); k++) {
            const double weight(config.mutator_config.weight_min +
                                (config.mutator_config.weight_max - config.mutator_config.weight_min) *
                                    random_.RandomNumber());
            genomes_.at(i).genes_.at(k).weight = weight;
        }
    }

    species_pool_.SortInSpecies(genomes_);
    innovation_ = genomes_.front().genes_.size();
}

std::vector<Network> Neat::GetNetworks() const {
    std::vector<Network> networks;

    for (const auto& genome : genomes_) {
        networks.push_back(Network(genome));
    }

    return networks;
}

void Neat::Evolve(std::vector<double> fitnesses, const uint& n_genotypes) {
    species_pool_.AdjustFitnesses(fitnesses, genomes_);
    species_pool_.Reproduce(genomes_, fitnesses, n_genotypes);
    mutator_.Mutate(genomes_, innovation_);
    species_pool_.SortInSpecies(genomes_);
}

SpeciesPool Neat::GetSpeciesPool() const { return species_pool_; }

std::vector<Genome> Neat::GetGenomes() const { return genomes_; }

uint Neat::GetInnovation() const { return innovation_; }

}  // namespace neat
