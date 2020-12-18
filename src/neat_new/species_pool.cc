#include "species_pool.h"

namespace neat {

SpeciesPool::SpeciesPool() {}

void SpeciesPool::SetConfig(const Config& config) { config_ = config; }

std::vector<SpeciesPool::Species> SpeciesPool::GetSpecies() const { return species_; }

void SpeciesPool::SortInSpecies(std::vector<Genome>& genomes) {
    for (auto& spec : species_) {
        spec.n_member = 0;
        spec.set_ref_genome = false;
    }

    if (!species_.size()) {
        Species spec;

        spec.ref_genome = genomes.front();
        spec.n_member = 0;
        genomes.front().species_id_ = 0;

        species_.push_back(spec);
    }

    for (auto& genome : genomes) {
        bool found(false);

        for (uint spec_id = 0; spec_id < species_.size(); spec_id++) {
            if (species_.at(spec_id).ref_genome.Distance(genome, config_.distance_coefficients) <
                config_.max_species_distance) {
                species_.at(spec_id).n_member++;
                species_.at(spec_id).SetRefGenome(genome);
                genome.species_id_ = spec_id;
                found = true;

                break;
            }
        }

        if (!found) {
            Species spec;

            spec.ref_genome = genome;
            spec.n_member = 1;
            genome.species_id_ = species_.size();

            species_.push_back(spec);
        }
    }
}

}  // namespace neat
