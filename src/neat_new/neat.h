#pragma once

#include "genome.h"
#include "mutator.h"
#include "network.h"
#include "species_pool.h"

namespace neat {
using uint = unsigned int;
class Neat {
   public:
    struct Config {
        Mutator::Config mutator_config;
        SpeciesPool::Config species_pool_config;
    };

    Neat();
    ~Neat() = default;

    void Initialize(const uint& n_sensor_nodes, const uint& n_output_nodes, const uint& n_genotypes,
                    const Config config);
    std::vector<Network> GetNetworks() const;
    void Evolve(std::vector<double> fitnesses);

    SpeciesPool GetSpeciesPool() const;
    std::vector<Genome> GetGenotypes() const;
    uint GetInnovation() const;

   private:
    std::vector<Genome> genomes_;
    Mutator mutator_;
    SpeciesPool species_pool_;
    Random random_;
    uint innovation_ = 0;
};
}  // namespace neat
