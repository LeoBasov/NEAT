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

    bool HasNode(const uint& node_id) const;

    double fitness_ = 0.0;
    std::vector<Gene> genes_;
    std::vector<uint> node_ids_;
};
}  // namespace NEAT
