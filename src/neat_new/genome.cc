#include "genome.h"

namespace neat {

Genome::Genome() {}

Genome::Genome(uint n_sensor_nodes, uint n_output_nodes) { Initialize(n_sensor_nodes, n_output_nodes); }

void Genome::Clear() {
    n_sensor_nodes_ = 0;
    n_output_nodes_ = 0;
    n_hidden_nodes_ = 0;
    nodes_.clear();
    genes_.clear();
}

uint Genome::Initialize(uint n_sensor_nodes, uint n_output_nodes) {
    uint innov(0);

    Clear();

    n_sensor_nodes_ = n_sensor_nodes;
    n_output_nodes_ = n_output_nodes;

    // initialize nodes
    for (uint i = 0; i < (n_sensor_nodes_ + n_output_nodes_ + 1); i++) {
        nodes_.push_back(i);
    }

    // initialize genes
    for (unsigned int i = 0; i < (n_sensor_nodes_ + 1); i++) {
        for (unsigned int o = (1 + n_sensor_nodes_); o < nodes_.size(); o++) {
            genes_.push_back(Gene(i, o, innov++));
        }
    }

    return --innov;
}

}  // namespace neat
