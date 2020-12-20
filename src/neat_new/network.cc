#include "network.h"

namespace neat {

Network::Network() {}

Network::Network(const Genome& genome) { Build(genome); }

void Network::Build(const Genome& genome) {
    matrix_ = GenomeToMatrix(genome);
    output_nodes_ = SetUpOutputNodes(genome);
    n_executions_ = genome.genes_.size();  // 3;
    n_const_nodes_ = genome.n_sensor_nodes_ + 1;
}

std::vector<double> Network::Execute(const std::vector<double>& input_vaules) {
    VectorXd vec(GenomeToVector(input_vaules, matrix_.rows()));

    for (uint i = 0; i < n_executions_; i++) {
        vec = matrix_ * vec;

        for (Eigen::Index i = n_const_nodes_; i < vec.rows(); i++) {
            vec(i) = utility::Sigmoid(vec(i), parameter_);
        }
    }

    return VectorToStd(vec, output_nodes_);
}

std::vector<size_t> Network::SetUpOutputNodes(const Genome& genome) {
    std::vector<size_t> output_nodes;

    for (uint i = genome.n_sensor_nodes_ + 1; i < (genome.n_output_nodes_ + genome.n_sensor_nodes_ + 1); i++) {
        output_nodes.push_back(i);
    }

    return output_nodes;
}

MatrixXd Network::GenomeToMatrix(const Genome& genome) {
    MatrixXd matrix(MatrixXd::Zero(genome.nodes_.size(), genome.nodes_.size()));
    const std::map<size_t, size_t> permutation_map(genome.GetNodePermuationMap());

    for (const auto& gene : genome.genes_) {
        if (gene.enabled) {
            const uint in_node(permutation_map.at(gene.in));
            const uint out_node(permutation_map.at(gene.out));

            matrix(out_node, in_node) = gene.weight;
        }
    }

    for (uint i = 0; i < 1 + genome.n_sensor_nodes_; i++) {
        matrix(i, i) = 1.0;
    }

    return matrix;
}

VectorXd Network::GenomeToVector(const std::vector<double>& input_vaules, const uint& n_nodes) {
    VectorXd vec(VectorXd::Zero(n_nodes));

    vec(0) = 1.0;

    for (size_t i = 0; i < input_vaules.size(); i++) {
        vec(i + 1) = input_vaules.at(i);
    }

    return vec;
}

std::vector<double> Network::VectorToStd(const VectorXd& vec, const std::vector<size_t>& output_nodes) {
    std::vector<double> ret_vals(output_nodes.size());

    for (uint i = 0; i < output_nodes.size(); i++) {
        ret_vals.at(i) = vec(output_nodes.at(i));
    }

    return ret_vals;
}

}  // namespace neat
