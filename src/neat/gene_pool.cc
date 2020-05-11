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

    for (uint out = output_nodes_.ofset; out < output_nodes_.ofset + output_nodes_.n_parts; out++) {
        for (uint in = input_nodes_.ofset; in < input_nodes_.ofset + input_nodes_.n_parts; in++) {
            genes_.push_back(Gene(in, out));
        }
    }
}

bool GenePool::AddNode(const uint& node_in, const uint& node_out) {
    if (nodes_.at(node_in).level < nodes_.at(node_out).level) {
        AdjustLevelsAbove(nodes_.at(node_in).level);
        nodes_.push_back(Node(nodes_.at(node_in).level + 1));
        hidden_nodes_.n_parts++;
        genes_.push_back(Gene(node_in, nodes_.size() - 1));
        genes_.push_back(Gene(nodes_.size() - 1, node_out));
        return true;
    } else {
        return false;
    }
}

bool GenePool::AddConnection(const uint& node_in, const uint& node_out) {
    if (nodes_.at(node_in).level < nodes_.at(node_out).level) {
        for (auto gene : genes_) {
            if (((gene.in == node_in) && (gene.out == node_out)) || ((gene.in == node_out) && (gene.out == node_in))) {
                return false;
            }
        }

        genes_.push_back(Gene(node_in, node_out));
        return true;
    } else {
        return false;
    }
}

void GenePool::AdjustLevelsAbove(const uint level) {
    for (uint i = 0; i < nodes_.size(); i++) {
        if (nodes_.at(i).level > level) {
            nodes_.at(i).level++;
        }
    }
}

}  // namespace NEAT
