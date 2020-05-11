#pragma once

#include <memory>

#include "gene_pool.h"
#include "network.h"
#include "phenotype.h"

namespace NEAT {
// class responsible for mutation and gene pool extension
class NEAT {
    NEAT();
    ~NEAT() = default;

    void Clear();
    void Execute(const std::vector<std::pair<VectorXd, VectorXd>> &input_outputs);

   private:
    GenePool gene_pool_;
    std::vector<Phenotype> phenotypes_;
    std::vector<Network> networks_;
};
}  // namespace NEAT
