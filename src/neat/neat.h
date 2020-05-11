#pragma once

#include <memory>

#include "../common/random.h"
#include "gene_pool.h"
#include "network.h"
#include "phenotype.h"

namespace NEAT {
// class responsible for mutation and gene pool extension
class NEAT {
    NEAT();
    ~NEAT() = default;

    void Clear();
    void Execute(const std::vector<std::pair<VectorXd, VectorXd>>& input_outputs);
    Phenotype Mate(const Phenotype& fitter_parent, const Phenotype& less_fit_parent);
    void ExecuteNetwork(const uint& network_id, const VectorXd& input, VectorXd& output) const;
    void BuildNetworks();

   private:
    Random random_;
    GenePool gene_pool_;
    std::vector<Phenotype> phenotypes_;
    std::vector<Network> networks_;
};
}  // namespace NEAT
