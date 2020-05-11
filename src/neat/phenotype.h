#pragma once

#include <vector>

namespace NEAT {
class Phenotype {
    using uint = unsigned int;

   public:
    struct Gene {
        uint id;
        bool enabled;
        double weight;
    };

    Phenotype();
    ~Phenotype() = default;

    double fitness_ = 0.0;
    std::vector<Gene> genes;
};
}  // namespace NEAT
