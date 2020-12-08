#pragma once

#include <vector>
#include <stdexcept>

namespace neat {
class GenePool {
   public:
    struct Gene {
        Gene(unsigned int in = 0, unsigned int out = 0) : in_node(in), out_node(out) {}

        unsigned int in_node = 0;
        unsigned int out_node = 0;
    };

    GenePool();
    ~GenePool() = default;

    void Clear();
    void Initialize(const unsigned int& n_sensor_nodes, const unsigned int& n_output_nodes);

    std::pair<unsigned int, unsigned int> AddNode(const unsigned int& gene_id);
    std::pair<bool, unsigned int> AddConnection(unsigned int in_node, unsigned int out_node);

    unsigned int GetNSensorNodes() const;
    unsigned int GetNOutputNodes() const;
    unsigned int GetNHiddenNodes() const;
    unsigned int GetNTotalNodes() const;  // This number is all nodes togehter + 1 for bias node
    std::vector<Gene> GetGenes() const;
    Gene GetGene(unsigned int id) const;

   private:
    // Bias node is allways node 0
    unsigned int n_sensor_nodes_ = 0;
    unsigned int n_output_nodes_ = 0;
    unsigned int n_hidden_nodes_ = 0;
    std::vector<Gene> genes_;
};
}  // namespace neat
