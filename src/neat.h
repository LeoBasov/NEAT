#pragma once

#include "gene_pool.h"
#include "phenotype.h"

namespace NEAT {
// class responsible for mutation and gene pool extension
class NEAT {
    NEAT();
    ~NEAT() = default;

   private:
    GenePool gene_pool_;
    std::vector<Phenotype> phenotypes_;
};
}  // namespace NEAT
