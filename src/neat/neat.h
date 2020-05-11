#pragma once

#include <memory>

#include "../common/random.h"
#include "gene_pool.h"
#include "phenotype.h"

namespace NEAT {
// class responsible for mutation and gene pool extension
class NEAT {
    NEAT();
    ~NEAT() = default;

    void Clear();
    void Execute(const std::vector<std::pair<VectorXd, VectorXd>>& input_outputs);
    Phenotype Mate(const Phenotype& fitter_parent, const Phenotype& less_fit_parent);
    void ExecutePhenotype(const Phenotype& phenotype, const VectorXd& input, VectorXd& output) const;

   private:
    Random random_;
    GenePool gene_pool_;
    std::vector<Phenotype> phenotypes_;
};
}  // namespace NEAT
