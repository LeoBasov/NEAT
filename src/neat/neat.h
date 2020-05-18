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
        // The number of input nodes must accound for an bias node
        uint n_input;
        uint n_output;
        uint n_phenotypes;
        double sigmoid_parameter = 1.0;
    };

    NEAT();
    ~NEAT() = default;

    void Clear();
    void Initialize(const Config& config);
    // The input vector mus contain a "BIAS 1"
    void Execute(const std::vector<std::pair<VectorXd, VectorXd>>& input_outputs);
    Phenotype Mate(const Phenotype& fitter_parent, const Phenotype& less_fit_parent);
    void ExecuteNetwork(const uint& network_id, const VectorXd& input, VectorXd& output) const;
    double ExecuteNode(const uint& network_id, const uint& node_id, const VectorXd& input) const;
    void BuildNetworks();
    double Sigmoid(const double& value, const double& paramters = 1.0) const;
    bool AddConnection(const uint& phenotype_id, const uint& in, const uint& out);
    bool AddNode(const uint& phenotype_id, const uint& in, const uint& out);
    bool SetWeight(const uint& phenotype_id, const uint& gene_id, const double& weight);
    bool ChangeActivation(const uint& phenotype_id, const uint& gene_id);

    Config config_;
    Random random_;
    GenePool gene_pool_;
    std::vector<Phenotype> phenotypes_;
    std::vector<Network> networks_;
};
}  // namespace NEAT
