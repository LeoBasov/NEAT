#pragma once

#include "../common/utility.h"
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
        double prob_mate = 0.75;
    };

    SpeciesPool();
    ~SpeciesPool() = default;

    void Clear();

    void SetConfig(const Config& config);
    std::vector<Species> GetSpecies() const;
    double GetTotalFitness() const;

    void SortInSpecies(std::vector<Genome>& genomes);
    void AdjustFitnesses(std::vector<double>& fitnesses, const std::vector<Genome>& genomes);
    void Reproduce(std::vector<Genome>& genotypes, const std::vector<double>& fitnesses, const uint& n_genotypes);
    void ReproduceSpecies(const Species& species, const std::vector<Genome>& genotypes,
                          std::vector<Genome>& new_genotypes, const uint& n_new_genotypes, const uint& species_id,
                          const double& prob_mate);
    void SortBySpecies(std::vector<Genome>& genotypes);
    void SortByFitness(const std::vector<double>& fitnesses, std::vector<Genome>& genotypes);

   private:
    std::vector<Species> species_;
    Config config_;
    double total_fitness_ = 0.0;
};
}  // namespace neat
