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
        double prob_weight_change = 0.9;
        double prob_new_weight = 0.1;
        double prob_new_node = 0.03;
        double prob_new_connection = 0.05;
        double prob_mate = 0.75;

        bool allow_self_connection = true, allow_recurring_connection = true;

        void Clear() {
            weight_range = {-10.0, 10.0};
            species_distance = 3.0;
            coeff1 = 1.0, coeff2 = 1.0, coeff3 = 0.4;
            sigmoid_parameter = 4.9;
            prob_weight_change = 0.9;
            prob_new_weight = 0.1;
            prob_new_node = 0.03;
            prob_new_connection = 0.05;
            prob_mate = 0.75;

            allow_self_connection = true, allow_recurring_connection = true;
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
    void SetSpecies(const std::vector<genome::Species>& species);
    void SetGenePool(const GenePool& gene_pool);

   private:
    Config config_;
    GenePool gene_pool_;
    std::vector<genome::Genotype> genotypes_;
    std::vector<genome::Species> species_;
    Random random_;
    uint n_genotypes_init_ = 0;
};
}  // namespace neat
