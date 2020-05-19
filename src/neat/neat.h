#pragma once

#include <memory>
#include <sstream>

#include "../common/random.h"
#include "gene_pool.h"
#include "network.h"
#include "phenotype.h"

namespace NEAT {
// class responsible for mutation and gene pool extension
class NEAT {
   public:
    struct Probabilities {
        double change_connection_weight = 0.8;
        double weight_pertubated = 0.1;
        double new_weight = 0.9;
        double new_node = 0.003;
        double new_connection = 0.05;
        double cross_over = 0.75;
        double interspecies = 0.001;
        double connection_activation = 0.01;
    };

    struct Config {
        // The number of input nodes must accound for an bias node
        uint n_input;
        uint n_output;
        uint n_phenotypes;
        uint max_imporf_free_itter = 15;
        double sigmoid_parameter = 4.9;
        double species_distance_threshold = 3.0;
        std::array<double, 3> species_distance_parameters{1.0, 1.0, 0.4};
        Probabilities probabilities;
    };

    struct Species {
        Species() {}
        Species(const Phenotype& phenotype) : ref_phenotype(phenotype) {}

        Phenotype ref_phenotype;
        std::vector<uint> phenotype_ids;
        uint n_offspring = 0;
        double total_adjusted_fitness = 0.0;

        void Clear() {
            phenotype_ids.clear();
            n_offspring = 0;
            total_adjusted_fitness = 0.0;
        }
    };

    NEAT();
    ~NEAT() = default;

    void Clear();
    void Initialize(const Config& config);
    // The input vector mus contain a "BIAS 1" => first in vection(0)
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
    double Distance(const Phenotype& first, const Phenotype& second, const std::array<double, 3>& parameters) const;
    void Speciate();
    void Reproduce();
    void Mutate();
    std::string Str(const uint& phenotype_id) const;

    Config config_;
    Random random_;
    GenePool gene_pool_;
    std::vector<Phenotype> phenotypes_;
    std::vector<Network> networks_;
    std::vector<Species> species_;
};
}  // namespace NEAT
