#include "gene_pool.h"

namespace NEAT {

GenePool::GenePool() {}

void GenePool::Clear() {
    input_nodes_.Clear();
    output_nodes_.Clear();
    hidden_nodes_.Clear();
    nodes_.clear();
    genes_.clear();
}

void GenePool::Initialize(const uint& n_input, const uint& n_output) {
    Clear();
    nodes_.resize(n_input + n_output);
    input_nodes_.n_parts = n_input;
    output_nodes_.ofset = n_input;
    output_nodes_.n_parts = n_output;
    hidden_nodes_.ofset = output_nodes_.ofset + output_nodes_.n_parts;

    for (uint i = 0; i < n_input; i++) {
        nodes_.push_back(Node(0));
    }

    for (uint i = 0; i < n_output; i++) {
        nodes_.push_back(Node(1));
    }
}

}  // namespace NEAT
