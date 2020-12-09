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
        double sigmoid_parameter = 4.9;

        void Clear() {
            weight_range = {-10.0, 10.0};
            species_distance = 3.0;
            coeff1 = 1.0, coeff2 = 1.0, coeff3 = 0.4;
            sigmoid_parameter = 4.9;
        };
    };

    NEAT();
    ~NEAT() = default;

    void Clear();

    // First to call
    void Initialize(const unsigned int& n_sensor_nodes, const unsigned int& n_output_nodes,
                    const unsigned int& n_genotypes, const Config config);

    // Second to call
    // Input values do NOT include bias node
    std::vector<std::vector<double> > ExecuteNetworks(const std::vector<double>& input_values) const;
    std::vector<double> ExecuteNetwork(const std::vector<double>& input_values, const uint& genotype_id) const;

    // Third to call
    void UpdateNetworks(std::vector<double> fitnesses);

    GenePool GetGenePool() const;
    std::vector<genome::Genotype> GetGenotypes() const;
    std::vector<genome::Species> GetSpecies() const;

    void SetGenotypes(const std::vector<genome::Genotype>& genotypes);

   private:
    Config config_;
    GenePool gene_pool_;
    std::vector<genome::Genotype> genotypes_;
    std::vector<genome::Species> species_;
    Random random_;
};
}  // namespace neat
