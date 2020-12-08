#pragma once

#include <stdexcept>
#include <vector>

#include "../common/random.h"
#include "gene_pool.h"
#include "genome.h"

namespace neat {
class NEAT {
   public:
    struct Config {
        std::pair<double, double> weight_range = {-10.0, 10.0};

        void Clear() { weight_range = {-10.0, 10.0}; };
    };

    NEAT();
    ~NEAT() = default;

    void Clear();
    void Initialize(const unsigned int& n_sensor_nodes, const unsigned int& n_output_nodes,
                    const unsigned int& n_genotypes, const Config config);

    GenePool GetGenePool() const;
    std::vector<genome::Genotype> GetGenotypes() const;

   private:
    Config config_;
    GenePool gene_pool_;
    std::vector<genome::Genotype> genotypes_;
    Random random_;
};
}  // namespace neat
