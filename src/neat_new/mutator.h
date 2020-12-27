#pragma once

#include <memory>

#include "genome.h"
#include "mutator_algorithms.h"

namespace neat {
using namespace mutator_algorithms;
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

    struct LastGene {
        enum Type { ADD_NODE, ADD_CONNECTION, NONE };

        Type type = NONE;
        uint in = 0, out = 0;
        std::pair<Genome::Gene, Genome::Gene> genes;
    };

   public:
    Mutator();
    ~Mutator() = default;

    void SetConfig(const Config& config);
    void SetrRandom(std::shared_ptr<Random> random);

    void Mutate(std::vector<Genome>& genomes, uint& innovation);
    void Mutate(Genome& genome, uint& innovation);
    std::pair<bool, uint> InLastGenes(const uint& in, const uint& out, LastGene::Type type) const;

   private:
    Config config_;
    std::shared_ptr<Random> random_ = std::make_shared<Random>();
    std::vector<LastGene> last_genes_;
};
}  // namespace neat
