#include "gene_pool.h"

namespace NEAT {

GenePool::GenePool() {}

void GenePool::Clear() {
    input_nodes_.Clear();
    output_nodes_.Clear();
    hidden_nodes_.Clear();
    nodes_.clear();
    genes_.clear();
    node_ofset_.clear();
    depth_ = 0;
}

void GenePool::Initialize(const uint& n_input, const uint& n_output) {
    Clear();
    input_nodes_.n_parts = n_input;
    output_nodes_.ofset = n_input;
    output_nodes_.n_parts = n_output;
    hidden_nodes_.ofset = output_nodes_.ofset + output_nodes_.n_parts;
    node_ofset_.push_back(0);
    node_ofset_.push_back(n_input);
    node_ofset_.push_back(n_input + n_output);
    depth_ = 2;

    for (uint i = 0; i < n_input; i++) {
        nodes_.push_back(Node(0));
    }

    for (uint i = 0; i < n_output; i++) {
        nodes_.push_back(Node(1));
    }

    for (uint out = output_nodes_.ofset; out < output_nodes_.ofset + output_nodes_.n_parts; out++) {
        for (uint in = input_nodes_.ofset; in < input_nodes_.ofset + input_nodes_.n_parts; in++) {
            genes_.push_back(Gene(in, out));
        }
    }
}

}  // namespace NEAT
