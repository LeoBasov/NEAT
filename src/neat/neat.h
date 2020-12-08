#pragma once

#include <stdexcept>
#include <vector>

#include "../common/random.h"
#include "gene_pool.h"
#include "genome.h"
#include "neat_algorithms.h"

namespace neat {
class NEAT {
   public:
    struct Config {
        std::pair<double, double> weight_range = {-10.0, 10.0};
        double species_distance = 3.0;
        double coeff1 = 1.0, coeff2 = 1.0, coeff3 = 0.4;

        void Clear() {
            weight_range = {-10.0, 10.0};
            species_distance = 3.0;
            coeff1 = 1.0, coeff2 = 1.0, coeff3 = 0.4;
        };
    };

    NEAT();
    ~NEAT() = default;

    void Clear();
    void Initialize(const unsigned int& n_sensor_nodes, const unsigned int& n_output_nodes,
                    const unsigned int& n_genotypes, const Config config);

    GenePool GetGenePool() const;
    std::vector<genome::Genotype> GetGenotypes() const;
    std::vector<genome::Species> GetSpecies() const;

   private:
    Config config_;
    GenePool gene_pool_;
    std::vector<genome::Genotype> genotypes_;
    std::vector<genome::Species> species_;
    Random random_;
};
}  // namespace neat
