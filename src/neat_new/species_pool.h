#pragma once

#include "genome.h"

namespace neat {
class SpeciesPool {
   public:
    struct Species {
        Genome ref_genome;
        uint n_member = 0;
        double total_fitness = 0.0;
        bool set_ref_genome = false;

        void SetRefGenome(const Genome& genome) {
            if (!set_ref_genome) {
                ref_genome = genome;
                set_ref_genome = true;
            }
        }
    };

    struct Config {
        std::array<double, 3> distance_coefficients{1.0, 1.0, 0.4};
        double max_species_distance = 3.0;
    };

    SpeciesPool();
    ~SpeciesPool() = default;

    void SetConfig(const Config& config);
    std::vector<Species> GetSpecies() const;
    double GetTotalFitness() const;

    void SortInSpecies(std::vector<Genome>& genomes);
    void AdjustFitnesses(std::vector<double>& fitnesses, const std::vector<Genome>& genomes);

   private:
    std::vector<Species> species_;
    Config config_;
    double total_fitness_ = 0.0;
};
}  // namespace neat
