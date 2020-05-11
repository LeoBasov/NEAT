#pragma once

#include <memory>

#include "../common/random.h"
#include "gene_pool.h"
#include "network.h"
#include "phenotype.h"

namespace NEAT {
// class responsible for mutation and gene pool extension
class NEAT {
   public:
    struct Config {
        uint n_input;
        uint n_output;
        uint n_phenotypes;
    };

    NEAT();
    ~NEAT() = default;

    void Clear();
    void Initialize(const Config& config);
    void Execute(const std::vector<std::pair<VectorXd, VectorXd>>& input_outputs);
    Phenotype Mate(const Phenotype& fitter_parent, const Phenotype& less_fit_parent);
    void ExecuteNetwork(const uint& network_id, const VectorXd& input, VectorXd& output) const;
    double ExecuteNode(const uint& network_id, const uint& node_id, const VectorXd& input) const;
    void BuildNetworks();
    double Sigmoid(const double& value) const;
    bool AddConnection(const uint& phenotype_id, const uint& in, const uint& out);
    bool AddNode(const uint& phenotype_id, const uint& in, const uint& out);

    Config config_;
    Random random_;
    GenePool gene_pool_;
    std::vector<Phenotype> phenotypes_;
    std::vector<Network> networks_;
};
}  // namespace NEAT
