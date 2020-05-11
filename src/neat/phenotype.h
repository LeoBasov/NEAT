#pragma once

#include <algorithm>
#include <vector>

namespace NEAT {
class Phenotype {
    using uint = unsigned int;

   public:
    struct Gene {
        uint id;
        bool enabled = true;
        double weight = 1.0;
    };

    Phenotype();
    ~Phenotype() = default;

    void AddGene(const Gene& gene);
    void AddGene(const uint& gene_id);
    bool AddGeneWithCheck(const uint& gene_id);

    double fitness_ = 0.0;
    std::vector<Gene> genes_;
};
}  // namespace NEAT
