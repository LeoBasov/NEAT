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

   private:
    GenePool gene_pool_;
    std::vector<Phenotype> phenotypes_;
    std::vector<std::shared_ptr<Network>> networks_;
};
}  // namespace NEAT
