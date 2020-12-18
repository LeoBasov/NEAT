#pragma once

#include "genome.h"

namespace neat {
class Mutator {
   public:
    struct Config {
        double prob_weight_change = 0.9;
        double prob_new_weight = 0.1;
        double prob_new_node = 0.03;
        double prob_new_connection = 0.05;

        double perturbation_fraction = 0.1;
        double weight_min = -10.0;
        double weight_max = 10.0;

        bool allow_self_connection = true;
        bool allow_recurring_connection = true;
    };

   public:
    Mutator();
    ~Mutator() = default;

    void SetConfig(const Config& config);
    void SetrRandom(const Random& random);

    void Mutate(Genome& genome, uint& innovation);
    void PertubateWeight(Genome& genome, Random& random, const uint& gene_id, const double& perturbation_fraq);

   private:
    Config config_;
    Random random_;
};
}  // namespace neat
