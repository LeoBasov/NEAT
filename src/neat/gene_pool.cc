#include "gene_pool.h"

namespace neat {

GenePool::GenePool() {}

void GenePool::Clear() {
    n_sensor_nodes_ = 0;
    n_output_nodes_ = 0;
    n_hidden_nodes_ = 0;
    genes_.clear();
}

void GenePool::Initialize(const unsigned int& n_sensor_nodes, const unsigned int& n_output_nodes) {
    Clear();

    n_sensor_nodes_ = n_sensor_nodes;
    n_output_nodes_ = n_output_nodes;

    for (unsigned int s = 0; s <= n_sensor_nodes_; s++) {
        for (unsigned int o = 0; o < n_output_nodes_; o++) {
            genes_.push_back(Gene(s, o));
        }
    }
}

unsigned int GenePool::GetNSensorNodes() const { return n_sensor_nodes_; }

unsigned int GenePool::GetNOutputNodes() const { return n_output_nodes_; }

unsigned int GenePool::GetNHiddenNodes() const { return n_hidden_nodes_; }

unsigned int GenePool::GetNTotalNodes() const { return n_sensor_nodes_ + n_output_nodes_ + n_hidden_nodes_; }

std::vector<GenePool::Gene> GenePool::GetGenes() const { return genes_; }

}  // namespace neat
