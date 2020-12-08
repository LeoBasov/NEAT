#pragma once

#include <stdexcept>
#include <vector>

#include "../common/random.h"
#include "gene_pool.h"

namespace neat {
class NEAT {
   public:
    struct Gene {
        Gene(unsigned int id, double weight = 1.0) : id(id), weight(weight) {}

        unsigned int id;
        double weight = 1.0;
        bool enabled = true;
    };

    struct Genotype {
        std::vector<Gene> genes;
    };

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
    std::vector<Genotype> GetGenotypes() const;

   private:
    Config config_;
    GenePool gene_pool_;
    std::vector<Genotype> genotypes_;
    Random random_;
};
}  // namespace neat
