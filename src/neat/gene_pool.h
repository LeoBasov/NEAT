#pragma once

#include <vector>

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

    unsigned int GetNSensorNodes() const;
    unsigned int GetNOutputNodes() const;
    unsigned int GetNHiddenNodes() const;
    unsigned int GetNTotalNodes() const;
    std::vector<Gene> GetGenes() const;

   private:
    // Bias node is allways node 0
    unsigned int n_sensor_nodes_ = 0;
    unsigned int n_output_nodes_ = 0;
    unsigned int n_hidden_nodes_ = 0;
    std::vector<Gene> genes_;
};
}  // namespace neat
