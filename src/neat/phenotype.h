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

        inline bool operator<(const Gene& rhs) { return this->id < rhs.id; }
        inline bool operator>(const Gene& rhs) { return this->id > rhs.id; }
        inline bool operator<=(const Gene& rhs) { return !(*this > rhs); }
        inline bool operator>=(const Gene& rhs) { return !(*this < rhs); }
    };

    Phenotype();
    ~Phenotype() = default;

    void AddGene(const Gene& gene);
    void AddGene(const uint& gene_id);
    bool AddGeneWithCheck(const uint& gene_id);

    double fitness_ = 0.0;
    std::vector<Gene> genes_;

    inline bool operator<(const Phenotype& rhs) { return this->fitness_ < rhs.fitness_; }
    inline bool operator>(const Phenotype& rhs) { return this->fitness_ > rhs.fitness_; }
    inline bool operator<=(const Phenotype& rhs) { return !(*this > rhs); }
    inline bool operator>=(const Phenotype& rhs) { return !(*this < rhs); }
};
}  // namespace NEAT
