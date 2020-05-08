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

   private:
    std::vector<Gene> genes;
};
}  // namespace NEAT
