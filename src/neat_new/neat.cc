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

void Neat::Evolve(std::vector<double>) {
    /*neat_algorithms::AdjustStagnationControll(fitnesses, best_fitness_, unimproved_counter_);
    neat_algorithms::AdjustedFitnesses(fitnesses, species_, genotypes_);

    if (unimproved_counter_ > config_.max_unimproved_iterations) {
        neat_algorithms::ReproduceBestSpecies(fitnesses, species_, genotypes_, n_genotypes_init_, config_.prob_mate, 2);

        best_fitness_ = 0.0;
        unimproved_counter_ = 0;
    } else {

    }

    neat_algorithms::Mutate(genotypes_, gene_pool_, config_.prob_weight_change, config_.prob_new_weight,
                            config_.prob_new_node, config_.prob_new_connection, config_.weight_range.first,
                            config_.weight_range.second, config_.allow_self_connection,
                            config_.allow_recurring_connection);*/
    mutator_.Mutate(genomes_, innovation_);
    species_pool_.SortInSpecies(genomes_);
}

SpeciesPool Neat::GetSpeciesPool() const { return species_pool_; }

std::vector<Genome> Neat::GetGenotypes() const { return genomes_; }

uint Neat::GetInnovation() const { return innovation_; }

}  // namespace neat
