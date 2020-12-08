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
        for (unsigned int o = (1 + n_sensor_nodes_); o < GetNTotalNodes(); o++) {
            genes_.push_back(Gene(s, o));
        }
    }
}

std::pair<unsigned int, unsigned int> GenePool::AddNode(const unsigned int& gene_id) {
    n_hidden_nodes_++;
    genes_.push_back(Gene(genes_.at(gene_id).in_node, GetNTotalNodes() - 1));
    genes_.push_back(Gene(GetNTotalNodes() - 1, genes_.at(gene_id).out_node));

    return {genes_.size() - 2, genes_.size() - 1};
}

std::pair<bool, unsigned int> GenePool::AddConnection(unsigned int in_node, unsigned int out_node) {
    // sensor nodes can not be out nodes of new connection
    if (out_node <= n_sensor_nodes_ || out_node >= GetNTotalNodes() || in_node >= GetNTotalNodes()) {
        return {false, 0};
    }

    for (size_t i = 0; i < genes_.size(); i++) {
        if (genes_.at(i).in_node == in_node && genes_.at(i).out_node == out_node) {
            return {true, i};
        }
    }

    genes_.push_back(Gene(in_node, out_node));

    return {true, genes_.size() - 1};
}

unsigned int GenePool::GetNSensorNodes() const { return n_sensor_nodes_; }

unsigned int GenePool::GetNOutputNodes() const { return n_output_nodes_; }

unsigned int GenePool::GetNHiddenNodes() const { return n_hidden_nodes_; }

unsigned int GenePool::GetNTotalNodes() const { return n_sensor_nodes_ + n_output_nodes_ + n_hidden_nodes_ + 1; }

std::vector<GenePool::Gene> GenePool::GetGenes() const { return genes_; }

GenePool::Gene GenePool::GetGene(unsigned int id) const { return genes_.at(id); }

}  // namespace neat
